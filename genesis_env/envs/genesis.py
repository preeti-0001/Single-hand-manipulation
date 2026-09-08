from __future__ import annotations

import time
from pathlib import Path

import genesis as gs
import numpy as np
from src.utils.common import (
    load_object_trajectory,
    load_robot_qpos_on_video_timeline,
    resolve_episode,
)
from src.utils.math_utils import matrix_to_wxyz, _to_numpy
import argparse
from .rewards import RewardModule
from .actor_critic import Actor, Critic
from .train_parallel import train_parallel_episode

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay an HRDexDB episode in Genesis."
    )

    parser.add_argument(
        "--hand",
        required=True,
        type=str,
        help="Hand name, e.g. allegro_v5",
    )

    parser.add_argument(
        "--object_name",
        required=True,
        type=str,
        help="Object name, e.g. apple",
    )

    parser.add_argument(
        "--scene",
        required=True,
        type=str,
        help="Scene ID, e.g. 4",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    HAND = args.hand
    OBJECT_NAME = args.object_name
    SCENE = args.scene

    # Resolve HRDexDB episode
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

    # Robot trajectory

    qpos, video_time, frame_ids, hand_dof, arm_dof = load_robot_qpos_on_video_timeline(
        ep.episode_root,
        ep.hand,
    )

    print("Robot qpos shape :", qpos.shape)
    print("Video time shape :", video_time.shape)
    print("Frame IDs shape  :", frame_ids.shape)

    # Object trajectory

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        HAND,
        OBJECT_NAME,
        SCENE,
    )

    # Determine replay length

    robot_frames = len(qpos)
    object_frames = len(object_poses)

    timeline_len = min(
        robot_frames,
        object_frames,
    )

    print("\n========== REPLAY ==========")
    print("Robot frames  :", robot_frames)
    print("Object frames :", object_frames)
    print("Replay frames :", timeline_len)
    print("============================\n")

    # Genesis

    gs.init(backend=gs.gpu, logging_level="warning")

    scene = gs.Scene(
        show_viewer=True,
        sim_options=gs.options.SimOptions(
            dt=1.0 / FPS,
            gravity=(0.0, 0.0, -9.81),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(0.7, 0.7, 0.45),
            camera_lookat=(0.0, 0.0, 0.15),
            camera_fov=40,
            res=(1280, 720),
        ),
    )

    #  Ground

    plane = scene.add_entity(gs.morphs.Plane())

    # Robot

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),
            pos=(0.0, 0.0, 0.0),
            fixed=True,
            # Required for your current URDF.
            recompute_inertia=True,
        )
    )

    # Object

    first_pose = object_poses[0]

    first_position = first_pose[:3, 3]

    first_quat = matrix_to_wxyz(first_pose[:3, :3])

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),
            pos=first_position,
            quat=first_quat,
            scale=1.0,
            # Kinematic replay false -> object will follow trajectory via given qpos/vel, not physics.
            fixed=False,
        ),
        material=gs.materials.Rigid(
            friction=1.5,
        ),
    )

    # Build
    scene.build()

    # Robot DOF check

    print("\n========== ROBOT ==========")
    print("Genesis qpos size :", robot.n_qs)
    print("Genesis DOFs      :", robot.n_dofs)
    print("HRDexDB qpos size :", qpos.shape[1])
    print("===========================\n")

    if robot.n_qs != qpos.shape[1]:

        raise RuntimeError(
            "\nDOF mismatch!\n"
            f"Genesis robot qpos : {robot.n_qs}\n"
            f"HRDexDB trajectory : {qpos.shape[1]}"
        )

    # Replay

    print(f"\nStarting replay " f"({timeline_len} frames @ {FPS} FPS)\n")

    frame_dt = 1.0 / FPS

    motors_dof_idx = [
        robot.get_joint(joint.name).dofs_idx_local[0]
        for i, joint in enumerate(robot.joints)
    ]

    demo_object_quaternions = []
    demo_object_trajectories = []
    demo_robot_trajectories = []
    demo_actions = []

    for frame in range(timeline_len):
        start_time = time.perf_counter()
        robot.set_dofs_position(
            qpos[frame],
            motors_dof_idx,
        )
        robot_keypoints = [
            robot.get_link(link.name).get_pos() for i, link in enumerate(robot.links)
        ]
        demo_robot_trajectories.append(robot_keypoints)
        demo_actions.append(robot.get_qpos()[motors_dof_idx])

        T = object_poses[frame]
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        quat_wxyz = matrix_to_wxyz(rotation_matrix)

        # Move object
        obj.set_pos(
            position,
            zero_velocity=True,
        )

        obj.set_quat(
            quat_wxyz,
            zero_velocity=True,
        )

        demo_object_trajectories.append(obj.get_pos())
        demo_object_quaternions.append(obj.get_quat())

        scene.step()

        elapsed = time.perf_counter() - start_time
        remaining = frame_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)

    print("\nReplay finished.")

    output_dir = ep.episode_root / "processed"

    contact_file = output_dir / "contact_tensor.npy"
    mask_file = output_dir / "validity_mask.npy"

    demo_contact_tensor = np.load(contact_file)
    demo_contact_validity = np.load(mask_file)

    reward_module = RewardModule(
        demo_contact_tensor,
        demo_object_trajectories,
        demo_object_quaternions,
        demo_robot_trajectories,
        demo_contact_validity,
        hand_dof,
        arm_dof,
        timeline_len,
        demo_actions,
    )

    actor = Actor(hand_dof + arm_dof)

    critic = Critic(hand_dof + arm_dof)

    states, actions, rewards, log_probs, values, total_reward = train_parallel_episode(
        reward_module,
        actor,
        critic,
        scene,
        robot,
        obj,
        timeline_len,
        motors_dof_idx,
        object_poses,
        env=128,
        num_episodes=100,
    )

    print("\n========== TRAINING RESULTS ==========")


if __name__ == "__main__":
    main()
