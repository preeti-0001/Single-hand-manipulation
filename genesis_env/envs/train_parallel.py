from __future__ import annotations

import time
from pathlib import Path
import torch

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
from .train_rl import train_one_episode

# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = Path("hrdexdb")
FPS = 30.0


def compute_gae(
    rewards,
    values,
    gamma=0.99,
    gae_lambda=0.95,
    device="cuda",
):
    rewards = torch.stack(
        [
            r if torch.is_tensor(r) else torch.tensor(r, dtype=torch.float32)
            for r in rewards
        ]
    ).to(device)

    values = torch.stack(values).squeeze(-1).to(device)

    # Terminal episode
    next_value = torch.tensor(0.0, device=device)

    advantages = torch.zeros_like(rewards)

    gae = 0.0

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_val = next_value
        else:
            next_val = values[t + 1]

        delta = rewards[t] + gamma * next_val - values[t]

        gae = delta + gamma * gae_lambda * gae

        advantages[t] = gae

    returns = advantages + values

    # Advantage normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    states,
    actions,
    old_log_probs,
    advantages,
    returns,
    ppo_epochs=10,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    states = torch.stack(states)
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)

    advantages = advantages.detach()
    returns = returns.detach()

    for _ in range(ppo_epochs):

        # ------------------------------------------------
        # Reconstruct state
        # ------------------------------------------------

        robot_qpos = states[:, :-7]
        object_pos = states[:, -7:-4]
        object_quat = states[:, -4:]

        # ------------------------------------------------
        # Current policy
        # ------------------------------------------------

        policy_dist = actor(
            robot_qpos,
            object_pos,
            object_quat,
        )

        new_log_probs = policy_dist.log_prob(actions).sum(dim=-1)

        entropy = policy_dist.entropy().sum(dim=-1).mean()

        # ------------------------------------------------
        # PPO probability ratio
        # ------------------------------------------------

        ratio = torch.exp(new_log_probs - old_log_probs)

        # ------------------------------------------------
        # Clipped objective
        # ------------------------------------------------

        unclipped = ratio * advantages

        clipped = (
            torch.clamp(
                ratio,
                1.0 - clip_eps,
                1.0 + clip_eps,
            )
            * advantages
        )

        actor_loss = -torch.min(
            unclipped,
            clipped,
        ).mean()

        # Entropy bonus
        actor_loss -= entropy_coef * entropy

        # ------------------------------------------------
        # Critic
        # ------------------------------------------------

        values = critic(
            robot_qpos,
            object_pos,
            object_quat,
        ).squeeze(-1)

        critic_loss = (returns - values).pow(2).mean()

        # ------------------------------------------------
        # Actor update
        # ------------------------------------------------

        actor_optimizer.zero_grad()
        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            actor.parameters(),
            0.5,
        )

        actor_optimizer.step()

        # ------------------------------------------------
        # Critic update
        # ------------------------------------------------

        critic_optimizer.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            critic.parameters(),
            0.5,
        )

        critic_optimizer.step()

    return (
        actor_loss.item(),
        critic_loss.item(),
        ratio.mean().item(),
    )


def train_parallel_episode(
    reward_module: RewardModule,
    actor: Actor,
    critic: Critic,
    scene,
    robot,
    obj,
    timeline_len,
    motors_dof_idx,
    object_poses,
    env: int = 128,
    num_episodes: int = 100,
):
    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=3e-4,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=1e-3,
    )
    for episode in range(num_episodes):

        (
            states,
            actions,
            rewards,
            old_log_probs,
            values,
            total_reward,
        ) = train_one_episode(
            reward_module,
            actor,
            critic,
            scene,
            robot,
            obj,
            timeline_len,
            motors_dof_idx,
            object_poses,
        )

        # ==============================================
        # GAE
        # ==============================================

        advantages, returns = compute_gae(rewards, values, gamma=0.99, gae_lambda=0.95)

        # ==============================================
        # PPO
        # ==============================================

        actor_loss, critic_loss, ratio = ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            states=states,
            actions=actions,
            old_log_probs=old_log_probs,
            advantages=advantages,
            returns=returns,
            ppo_epochs=10,
            clip_eps=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )

        print(
            f"Episode {episode} | "
            f"Reward {total_reward:.3f} | "
            f"Actor {actor_loss:.4f} | "
            f"Critic {critic_loss:.4f} | "
            f"Advantage stats: "
            f"mean={advantages.mean().item():.4f}, "
            f"std={advantages.std().item():.4f}, "
            f"min={advantages.min().item():.4f}, "
            f"max={advantages.max().item():.4f} | "
            f"Ratio {ratio:.3f}"
        )
