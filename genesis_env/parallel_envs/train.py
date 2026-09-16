from __future__ import annotations

import argparse
import time
from pathlib import Path

import genesis as gs
import numpy as np
import torch

from src.utils.common import (
    load_object_trajectory,
    load_robot_qpos_on_video_timeline,
    resolve_episode,
)
from src.utils.math_utils import matrix_to_wxyz

from .actor_critic import Actor, Critic
from .ppo import train_parallel_episode as train_parallel_episode_without_voc
from .voc.ppo import train_parallel_episode as train_parallel_episode_with_voc
from .rewards import RewardModule

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0
NUM_ENVS = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO with 128 parallel manipulation environments."
    )

    parser.add_argument(
        "--hand",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--object_name",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--scene",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--num_iterations",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--num_envs",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--action_scale",
        type=float,
        default=0.1,
        help="Maximum approximate joint delta in radians per policy step.",
    )

    parser.add_argument(
        "--without_voc",
        action="store_true",
        help="Disable the virtual object controller.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    HAND = args.hand
    OBJECT_NAME = args.object_name
    SCENE = args.scene

    NUM_PARALLEL_ENVS = args.num_envs

    # ============================================================
    # DATASET
    # ============================================================

    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=HAND,
        object_name=OBJECT_NAME,
        scene=SCENE,
    )

    print("\n========== HRDexDB EPISODE ==========")
    print("Episode root :", ep.episode_root)
    print("Hand         :", ep.hand)
    print("Object       :", ep.object_name)
    print("Robot URDF   :", ep.robot_urdf)
    print("Object mesh  :", ep.object_mesh)
    print("=====================================\n")

    # ============================================================
    # DEMONSTRATION
    # ============================================================

    qpos, video_time, frame_ids, hand_dof, arm_dof = load_robot_qpos_on_video_timeline(
        ep.episode_root,
        ep.hand,
    )

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        HAND,
        OBJECT_NAME,
        SCENE,
    )

    timeline_len = min(
        len(qpos),
        len(object_poses),
    )

    print("Robot qpos shape :", qpos.shape)
    print("Object frames    :", len(object_poses))
    print("Timeline length  :", timeline_len)

    # ============================================================
    # GENESIS
    # ============================================================

    gs.init(
        backend=gs.gpu,
        logging_level="warning",
    )

    scene = gs.Scene(
        # Headless is important for RL throughput.
        show_viewer=False,
        sim_options=gs.options.SimOptions(
            dt=1.0 / FPS,
            gravity=(0.0, 0.0, -9.81),
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            recompute_inertia=True,
        )
    )

    first_pose = object_poses[0]

    first_position = first_pose[:3, 3]

    first_quat = matrix_to_wxyz(first_pose[:3, :3])

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),
            pos=first_position,
            quat=first_quat,
            scale=1.0,
            fixed=False,
        ),
        material=gs.materials.Rigid(
            friction=1.5,
        ),
    )

    # ============================================================
    # 128 PARALLEL WORLDS
    # ============================================================

    scene.build(
        n_envs=NUM_PARALLEL_ENVS,
        env_spacing=(1.0, 1.0),
    )

    print(f"\nCreated {NUM_PARALLEL_ENVS} parallel worlds.")

    # ============================================================
    # DOFS
    # ============================================================

    motors_dof_idx = [
        robot.get_joint(joint.name).dofs_idx_local[0] for joint in robot.joints
    ]

    robot_dof = len(motors_dof_idx)

    print("\n========== ROBOT ==========")
    print("Genesis qpos shape :", robot.get_qpos().shape)
    print("Genesis DOFs       :", robot.n_dofs)
    print("HRDexDB qpos shape :", qpos.shape)
    print("Robot action DOFs  :", robot_dof)
    print("===========================\n")

    if robot_dof != qpos.shape[1]:
        raise RuntimeError(
            "Action DOF mismatch: " f"Genesis={robot_dof}, HRDexDB={qpos.shape[1]}"
        )

    device = robot.get_qpos().device
    dtype = robot.get_qpos().dtype

    # ============================================================
    # DEMO TENSORS
    # ============================================================

    demo_robot_qpos = torch.as_tensor(
        qpos[:timeline_len],
        device=device,
        dtype=dtype,
    )

    demo_object_positions = torch.as_tensor(
        np.asarray([T[:3, 3] for T in object_poses[:timeline_len]]),
        device=device,
        dtype=dtype,
    )

    demo_object_quaternions = torch.as_tensor(
        np.asarray([matrix_to_wxyz(T[:3, :3]) for T in object_poses[:timeline_len]]),
        device=device,
        dtype=dtype,
    )

    # ============================================================
    # DEMO ROBOT KEYPOINTS
    #
    # We use the same Genesis model to obtain the target link
    # trajectory. No physics rollout is used to create the target.
    # ============================================================

    demo_robot_keypoints = []

    for frame in range(timeline_len):

        robot.set_dofs_position(
            demo_robot_qpos[frame].unsqueeze(0).expand(NUM_PARALLEL_ENVS, -1),
            motors_dof_idx,
            zero_velocity=True,
        )

        links_pos = robot.get_links_pos()

        # Store only one copy because all 128 environments are
        # initialized identically during demonstration extraction.
        demo_robot_keypoints.append(links_pos[0].detach())

    # ============================================================
    # CONTACT DEMONSTRATION
    # ============================================================

    output_dir = ep.episode_root / "processed"

    contact_file = output_dir / "contact_tensor.npy"

    mask_file = output_dir / "validity_mask.npy"
    demo_ideal_grasps = np.load(output_dir / "ideal_grasps.npy", allow_pickle=True)

    demo_contact_tensor = np.load(contact_file)[:timeline_len]

    demo_contact_validity = np.load(mask_file)[:timeline_len]

    # ============================================================
    # REWARD
    # ============================================================

    reward_module = RewardModule(
        demo_contact_tensor=demo_contact_tensor,
        demo_contact_validity=demo_contact_validity,
        demo_robot_keypoints=demo_robot_keypoints,
        demo_robot_qpos=demo_robot_qpos,
        demo_object_trajectories=demo_object_positions,
        demo_object_quaternions=demo_object_quaternions,
        demo_ideal_grasps=demo_ideal_grasps,
        # Reward design:
        #
        # Robot trajectory and object trajectory are primary.
        # Contact and BC are auxiliary.
        beta_imitation=0.1,
        beta_contact=10.0,
        beta_position=0.1,
        beta_rotation=0.5,
        beta_bc=0.2,
        beta_grasp_position=20.0,
        beta_grasp_qpos=1.0,
        lambda_grasp=3.0,
        lambda_task=5.0,
        lambda_imitation=50,
        lambda_contact=3.0,
        lambda_bc=0.1,
        contact_dmax=0.05,
    )

    # ============================================================
    # ACTOR / CRITIC
    # ============================================================

    actor = Actor(
        robot_dof=robot_dof,
    ).to(device)

    critic = Critic(
        robot_dof=robot_dof,
    ).to(device)

    print(
        "Actor parameters :",
        sum(p.numel() for p in actor.parameters()),
    )

    print(
        "Critic parameters:",
        sum(p.numel() for p in critic.parameters()),
    )

    # ============================================================
    # TRAIN
    # ============================================================

    print("\n========== PPO TRAINING ==========")
    if args.without_voc:
        train_parallel_episode = train_parallel_episode_without_voc
    else:
        train_parallel_episode = train_parallel_episode_with_voc

    train_parallel_episode(
        reward_module=reward_module,
        actor=actor,
        critic=critic,
        scene=scene,
        robot=robot,
        obj=obj,
        timeline_len=timeline_len,
        motors_dof_idx=motors_dof_idx,
        demo_robot_qpos=demo_robot_qpos,
        demo_object_positions=demo_object_positions,
        demo_object_quaternions=demo_object_quaternions,
        num_episodes=args.num_iterations,
        action_scale=args.action_scale,
    )

    print("\n========== TRAINING COMPLETE ==========")


if __name__ == "__main__":
    main()
