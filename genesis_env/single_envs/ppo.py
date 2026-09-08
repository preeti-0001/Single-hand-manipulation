from __future__ import annotations

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
CHECKPOINT_DIR = Path("logs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


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
    action_scale = 0.1

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
        delta_q = policy_action * action_scale

        target_qpos = robot_qpos + delta_q

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
    best_reward = -float("inf")
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
        if total_reward > best_reward:
            best_reward = total_reward

            torch.save(
                {
                    "episode": episode,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "actor_optimizer_state_dict": actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict": critic_optimizer.state_dict(),
                    "reward": total_reward,
                },
                CHECKPOINT_DIR / "best_ppo.pt",
            )

            print(f"New best policy saved! " f"Reward = {total_reward:.3f}")

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
