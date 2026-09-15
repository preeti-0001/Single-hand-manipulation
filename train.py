from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import genesis as gs
from src.utils.math_utils import matrix_to_wxyz

from src.utils.common import (
    resolve_episode,
    load_robot_qpos_on_video_timeline,
    load_object_trajectory,
    load_robot_actions
)

from src.ppo.actor import Actor
from src.ppo.critic import Critic
from src.ppo.ppo import PPOConfig, train
from src.ppo.base_env import HRDexEnv
from src.ppo.rewards import RewardModule


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")

HAND = "allegro_v5"
OBJECT_NAME = "apple"
SCENE = "4"

FPS = 30.0

NUM_ITERATIONS = 1000

# PPO action scale.
# PPO produces action in [-1, 1].
# Actual joint target change = action * ACTION_SCALE.
ACTION_SCALE = 0.02

# PPO control DOFs.
# None = first action_dim robot DOFs.
CONTROL_DOFS = None

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




# ============================================================
# CONTACT DATA
# ============================================================

def load_contact_data(episode_root: Path):
    """
    Adapt these filenames if your optimized contact
    preprocessing saves them elsewhere.
    """

    contact_file = episode_root / "processed/contact_tensor.npy"
    validity_file = episode_root / "processed/validity_mask.npy"

    if not contact_file.exists():
        raise FileNotFoundError(
            f"Contact tensor not found:\n{contact_file}"
        )

    if not validity_file.exists():
        raise FileNotFoundError(
            f"Validity mask not found:\n{validity_file}"
        )

    contact_tensor = np.load(contact_file)
    validity_mask = np.load(validity_file)

    print("\n========== CONTACT DATA ==========")
    print("Contact tensor :", contact_tensor.shape)
    print("Validity mask  :", validity_mask.shape)
    print("==================================")

    return contact_tensor, validity_mask

# ============================================================
# BUILD GENESIS
# ============================================================

def build_genesis(ep, object_poses):

    print("\n========== GENESIS ==========")

    gs.init(
        backend=gs.gpu
        if torch.cuda.is_available()
        else gs.cpu
    )

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

    # --------------------------------------------------------
    # Ground
    # --------------------------------------------------------

    scene.add_entity(
        gs.morphs.Plane()
    )

    # --------------------------------------------------------
    # Robot
    # --------------------------------------------------------

    robot = scene.add_entity(
        gs.morphs.URDF(
            file=str(ep.robot_urdf),

            pos=(0.0, 0.0, 0.0),

            fixed=True,

            recompute_inertia=True,
        )
    )

    # --------------------------------------------------------
    # Object
    # --------------------------------------------------------

    first_pose = object_poses[0]

    first_position = first_pose[:3, 3]

    first_quat = matrix_to_wxyz(
        first_pose[:3, :3]
    )

    # IMPORTANT:
    #
    # For PPO training the object should NOT be replayed
    # frame-by-frame.
    #
    # It starts here and then Genesis physics controls it.
    #

    obj = scene.add_entity(
        gs.morphs.Mesh(
            file=str(ep.object_mesh),

            pos=first_position,

            quat=first_quat,

            scale=1.0,

            fixed=False,
        )
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    scene.build()

    print("Robot qpos:", robot.n_qs)
    print("Robot DOFs:", robot.n_dofs)

    print("============================")

    return scene, robot, obj


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("HRDexDB + Genesis + PPO")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    # ========================================================
    # Resolve episode
    # ========================================================

    ep = resolve_episode(
        dataset_root=DATASET_ROOT,
        hand=HAND,
        object_name=OBJECT_NAME,
        scene=SCENE,
    )

    print("\n========== EPISODE ==========")

    print("Episode root :", ep.episode_root)
    print("Hand         :", ep.hand)
    print("Object       :", ep.object_name)
    print("Robot URDF   :", ep.robot_urdf)
    print("Object mesh  :", ep.object_mesh)

    print("=============================")

    # ========================================================
    # Robot demonstration
    # ========================================================

    demo_qpos, video_time, frame_ids, hand_dof, arm_dof = (
        load_robot_qpos_on_video_timeline(
            ep.episode_root,
            ep.hand,
        )
    )

    print("\n========== ROBOT DEMO ==========")
    print("qpos       :", demo_qpos.shape)
    print("video_time :", video_time.shape)
    print("frame_ids  :", frame_ids.shape)

    # ========================================================
    # Object trajectory
    # ========================================================

    object_poses = load_object_trajectory(
        DATASET_ROOT,
        HAND,
        OBJECT_NAME,
        SCENE,
    )

    print("\n========== OBJECT DEMO ==========")
    print("object poses:", object_poses.shape)

    # ========================================================
    # Contact data
    # ========================================================

    contact_tensor, validity_mask = load_contact_data(
        ep.episode_root
    )

    # ========================================================
    # Demonstration actions
    # ========================================================

    demo_action, demo_action_time = load_robot_actions(
        ep.episode_root,
        ep.hand
    )

    # ========================================================
    # Align episode length
    # ========================================================

    episode_length = min(
        len(demo_qpos),
        len(object_poses),
        len(contact_tensor),
        len(validity_mask),
        len(demo_action),
    )

    demo_qpos = demo_qpos[:episode_length]

    object_poses = object_poses[:episode_length]

    contact_tensor = contact_tensor[:episode_length]

    validity_mask = validity_mask[:episode_length]

    demo_action = demo_action[:episode_length]

    print("\n========== ALIGNED DATA ==========")
    print("Episode length :", episode_length)
    print("qpos           :", demo_qpos.shape)
    print("actions        :", demo_action.shape)
    print("contacts       :", contact_tensor.shape)
    print("validity       :", validity_mask.shape)
    print("object poses   :", object_poses.shape)
    print("===================================")

    # ========================================================
    # Check action dimension
    # ========================================================

    action_dim = demo_action.shape[-1]

    if action_dim > demo_qpos.shape[-1]:

        raise RuntimeError(
            "\nAction dimension is larger than robot qpos.\n"
            f"Action dim : {action_dim}\n"
            f"Robot qpos : {demo_qpos.shape[-1]}"
        )

    # ========================================================
    # Genesis
    # ========================================================

    scene, robot, obj = build_genesis(
        ep,
        object_poses,
    )

    # ========================================================
    # Control DOFs
    # ========================================================

    if CONTROL_DOFS is None:

        control_dofs = list(
            range(action_dim)
        )

    else:

        control_dofs = CONTROL_DOFS

    if max(control_dofs) >= robot.n_dofs:

        raise RuntimeError(
            "\nInvalid control DOFs.\n"
            f"Robot DOFs : {robot.n_dofs}\n"
            f"Control DOFs : {control_dofs}"
        )

    # ========================================================
    # Target
    # ========================================================

    #
    # Initial simple target:
    #
    # final demonstration object position.
    #
    # Later this should come from your task definition.
    #

    target_pos = object_poses[-1, :3, 3]

    # Don't use target quaternion yet.
    #
    # The current task reward depends on repository-specific
    # rotation representation.
    #

    target_quat = None

    # ========================================================
    # Reward
    # ========================================================

    reward_module = RewardModule(
        beta_contact=5.0,

        beta_position=5.0,

        beta_rotation=2.0,

        beta_angle=1.0,

        beta_bc=2.0,

        lambda_task=1.0,

        lambda_contact=1.0,

        lambda_imitation=0.0,

        lambda_bc=0.25,

        contact_dmax=0.05,
    )

    # ========================================================
    # Environment
    # ========================================================

    env = HRDexEnv(

        scene=scene,

        robot=robot,

        obj=obj,

        demo_qpos=demo_qpos,

        demo_action=demo_action,

        contact_tensor=contact_tensor,

        validity_mask=validity_mask,

        target_pos=target_pos,

        target_quat=target_quat,

        action_scale=ACTION_SCALE,

        control_dofs=control_dofs,

        reward_module=reward_module,

        max_steps=episode_length,

        device=DEVICE,
    )
    env.reset()

    print("\n========== ENVIRONMENT ==========")
    print("Observation dim :", env.obs_dim)
    print("Action dim      :", env.action_dim)
    print("Episode length  :", env.demo_length)
    print("Action scale    :", ACTION_SCALE)
    print("Control DOFs    :", control_dofs)
    print("=================================")


    # ========================================================
    # Actor
    # ========================================================

    actor = Actor(
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        hidden_dim=256,
    ).to(DEVICE)

    # ========================================================
    # Critic
    # ========================================================

    critic = Critic(
        obs_dim=env.obs_dim,
        hidden_dim=256,
    ).to(DEVICE)

    print("\n========== NETWORKS ==========")

    print(
        "Actor parameters :",
        sum(p.numel() for p in actor.parameters()),
    )

    print(
        "Critic parameters:",
        sum(p.numel() for p in critic.parameters()),
    )

    print("==============================")

    # ========================================================
    # PPO configuration
    # ========================================================

    ppo_config = PPOConfig(

        gamma=0.99,

        gae_lambda=0.95,

        clip_eps=0.2,

        value_coef=0.5,

        entropy_coef=0.01,

        learning_rate=3e-4,

        value_learning_rate=1e-3,

        update_epochs=10,

        minibatch_size=64,

        max_grad_norm=0.5,

        normalize_advantage=True,
    )

    # ========================================================
    # Train
    # ========================================================

    print("\n")
    print("=" * 70)
    print("STARTING PPO TRAINING")
    print("=" * 70)

    ppo = train(

        env=env,

        actor=actor,

        critic=critic,

        num_iterations=NUM_ITERATIONS,

        max_steps=episode_length,

        config=ppo_config,
    )

    # ========================================================
    # Save
    # ========================================================

    checkpoint = {

        "actor": actor.state_dict(),

        "critic": critic.state_dict(),

        "actor_optimizer": ppo.actor_optimizer.state_dict(),

        "critic_optimizer": ppo.critic_optimizer.state_dict(),

        "obs_dim": env.obs_dim,

        "action_dim": env.action_dim,

        "action_scale": ACTION_SCALE,

        "control_dofs": control_dofs,

        "hand": HAND,

        "object": OBJECT_NAME,

        "scene": SCENE,
    }

    save_path = (
        Path("logs/checkpoints")
        / f"ppo_{HAND}_{OBJECT_NAME}_{SCENE}.pt"
    )

    save_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        checkpoint,
        save_path,
    )

    print("\n")
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print("Checkpoint:", save_path)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()