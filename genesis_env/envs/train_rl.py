from __future__ import annotations

import time
from pathlib import Path
import torch

from src.utils.math_utils import matrix_to_wxyz
from .rewards import RewardModule
from .actor_critic import Actor, Critic

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


def train_one_episode(
    reward_module: RewardModule,
    actor: Actor,
    critic: Critic,
    scene,
    robot,
    obj,
    timeline_len,
    motors_dof_idx,
    object_poses,
):

    print(f"\nStarting training ({timeline_len} frames)\n")

    total_reward = 0.0

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    for frame in range(timeline_len):

        # ==================================================
        # CURRENT STATE
        # ==================================================

        robot_qpos = robot.get_qpos()

        object_pos = obj.get_pos()
        object_quat = obj.get_quat()

        state = torch.cat(
            [
                robot_qpos,
                object_pos,
                object_quat,
            ]
        )

        # ==================================================
        # ACTOR
        # ==================================================

        policy_dist = actor(
            robot_qpos,
            object_pos,
            object_quat,
        )

        policy_action = policy_dist.sample()

        log_prob = policy_dist.log_prob(policy_action).sum()

        # ==================================================
        # CRITIC
        # ==================================================

        value = critic(
            robot_qpos,
            object_pos,
            object_quat,
        )

        # ==================================================
        # EXECUTE ACTION
        # ==================================================

        robot.set_dofs_position(
            policy_action.detach().cpu().numpy(),
            motors_dof_idx,
        )

        # ==================================================
        # DEMO OBJECT POSE
        # ==================================================

        T = object_poses[frame]

        position = T[:3, 3]

        rotation_matrix = T[:3, :3]

        quat_wxyz = matrix_to_wxyz(rotation_matrix)

        obj.set_pos(
            position,
            zero_velocity=True,
        )

        obj.set_quat(
            quat_wxyz,
            zero_velocity=True,
        )

        # ==================================================
        # SIMULATION
        # ==================================================

        scene.step()

        # ==================================================
        # OBSERVE RESULT
        # ==================================================

        current_keypoints = [
            robot.get_link(link.name).get_pos() for link in robot.links
        ]

        current_contacts = robot.get_contacts(with_entity=obj)

        object_pos_next = obj.get_pos()
        object_quat_next = obj.get_quat()

        # ==================================================
        # REWARD
        # ==================================================

        reward = reward_module.compute_total_reward(
            current_keypoints,
            current_contacts,
            object_pos_next,
            object_quat_next,
            policy_action,
            frame,
        )

        total_reward += reward

        # ==================================================
        # STORE PPO DATA
        # ==================================================

        states.append(state.detach())
        actions.append(policy_action.detach())
        rewards.append(reward)
        log_probs.append(log_prob.detach())
        values.append(value.detach())

    return (
        states,
        actions,
        rewards,
        log_probs,
        values,
        total_reward,
    )
