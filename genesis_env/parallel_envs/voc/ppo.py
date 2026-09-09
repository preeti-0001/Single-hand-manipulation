from __future__ import annotations

from pathlib import Path

import torch

from .actor_critic import Actor, Critic
from .rewards import RewardModule

from datetime import datetime


CHECKPOINT_DIR = Path("logs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _make_state(
    robot_qpos,
    object_pos,
    object_quat,
    target_robot_qpos,
    target_object_pos,
    target_object_quat,
):
    return torch.cat(
        [
            robot_qpos,
            object_pos,
            object_quat,
            target_robot_qpos,
            target_object_pos,
            target_object_quat,
        ],
        dim=-1,
    )


def _quaternion_error_wxyz(target_quat, current_quat):
    """Shortest-arc quaternion error as a 3D axis-angle-like vector."""
    target_quat = target_quat / (
        target_quat.norm(dim=-1, keepdim=True) + 1e-8
    )
    current_quat = current_quat / (
        current_quat.norm(dim=-1, keepdim=True) + 1e-8
    )

    tw, tx, ty, tz = target_quat.unbind(dim=-1)
    cw, cx, cy, cz = current_quat.unbind(dim=-1)

    # q_error = q_target * conjugate(q_current)
    ew = tw * cw + tx * cx + ty * cy + tz * cz
    ex = -tw * cx + tx * cw - ty * cz + tz * cy
    ey = -tw * cy + tx * cz + ty * cw - tz * cx
    ez = -tw * cz - tx * cy + ty * cx + tz * cw

    sign = torch.where(ew < 0.0, -1.0, 1.0)
    return 2.0 * sign.unsqueeze(-1) * torch.stack([ex, ey, ez], dim=-1)


def _apply_virtual_object_controller(
    obj,
    scene,
    current_pos,
    current_quat,
    current_lin_vel,
    target_pos,
    target_quat,
    target_lin_vel,
    voc_strength,
    kp_pos=40.0,
    kv_pos=8.0,
    kp_rot=8.0,
    kv_rot=1.5,
    max_force=25.0,
    max_torque=5.0,
):
    """
    Apply one-step virtual spring-damper control to the free object.

    VOC is privileged simulation assistance: it pulls the object toward
    the demonstration target but does not teleport the object.
    """
    pos_error = target_pos - current_pos
    vel_error = target_lin_vel - current_lin_vel

    force = (
        kp_pos * pos_error
        + kv_pos * vel_error
    )

    rot_error = _quaternion_error_wxyz(
        target_quat,
        current_quat,
    )

    # For a rigid object, angular velocity is available from the root link.
    current_ang_vel = obj.get_links_ang(
        links_idx_local=[0]
    )[:, 0, :]

    torque = (
        kp_rot * rot_error
        - kv_rot * current_ang_vel
    )

    force_norm = force.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    torque_norm = torque.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    force = force * torch.clamp(
        max_force / force_norm,
        max=1.0,
    )
    torque = torque * torch.clamp(
        max_torque / torque_norm,
        max=1.0,
    )

    strength = voc_strength.unsqueeze(-1)

    force = strength * force
    torque = strength * torque

    # Genesis external wrench API expects [N, n_links, 3].
    # obj.apply_links_external_wrench(
    #     force=force.unsqueeze(1),
    #     torque=torque.unsqueeze(1),
    #     links_idx_local=[0],
    # )
    rigid_solver = scene.sim.rigid_solver

    obj_link_idx = obj.base_link.idx

    rigid_solver.apply_links_external_force(
    force=force.unsqueeze(1),
    links_idx=[obj_link_idx],
)

    rigid_solver.apply_links_external_torque(
    torque=torque.unsqueeze(1),
    links_idx=[obj_link_idx],
)


def train_one_episode(
    reward_module: RewardModule,
    actor: Actor,
    critic: Critic,
    scene,
    robot,
    obj,
    timeline_len,
    motors_dof_idx,
    demo_robot_qpos,
    demo_object_positions,
    demo_object_quaternions,
    robot_q_min,
    robot_q_max,
    action_scale=0.1,
    voc_strength=None,
    voc_contact_ema=None,
    voc_decay_rate=0.02,
    voc_contact_threshold=0.65,
    voc_min_strength=0.0,
):
    """
    Run one trajectory in ALL Genesis environments simultaneously.

    Shapes:
        robot_qpos     [N, D]
        action         [N, D]
        reward         [N]

    N = number of parallel worlds.
    """

    num_envs = robot.get_qpos().shape[0]
    device = robot.get_qpos().device
    dtype = robot.get_qpos().dtype

    # ------------------------------------------------------------
    # Reset every environment to the same demonstration start.
    # ------------------------------------------------------------

    q0 = demo_robot_qpos[0].to(
        device=device,
        dtype=dtype,
    )

    robot.set_dofs_position(
        q0.unsqueeze(0).expand(num_envs, -1),
        motors_dof_idx,
        zero_velocity=True,
    )

    obj.set_pos(
        demo_object_positions[0].to(device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(num_envs, -1),
        zero_velocity=True,
    )

    obj.set_quat(
        demo_object_quaternions[0].to(device=device, dtype=dtype)
        .unsqueeze(0)
        .expand(num_envs, -1),
        zero_velocity=True,
    )

    states = []
    actions = []
    rewards = []
    log_probs = []
    values = []

    total_reward = torch.zeros(
        num_envs,
        device=device,
        dtype=dtype,
    )

    if voc_strength is None:
        voc_strength = torch.ones(
            num_envs,
            device=device,
            dtype=dtype,
        )

    if voc_contact_ema is None:
        voc_contact_ema = torch.zeros_like(voc_strength)

    for frame in range(timeline_len - 1):

        # ========================================================
        # CURRENT STATE
        # ========================================================

        robot_qpos = robot.get_qpos(
            qs_idx_local=motors_dof_idx
        )

        object_pos = obj.get_pos()
        object_quat = obj.get_quat()

        # ========================================================
        # DESIRED STATE
        #
        # The policy MUST see the target. Otherwise the same
        # current state can occur at different trajectory phases
        # but require different actions.
        # ========================================================

        target_frame = frame + 1

        target_robot_qpos = demo_robot_qpos[
            target_frame
        ].to(device=device, dtype=dtype)

        target_object_pos = demo_object_positions[
            target_frame
        ].to(device=device, dtype=dtype)

        target_object_quat = demo_object_quaternions[
            target_frame
        ].to(device=device, dtype=dtype)

        target_robot_qpos_batch = (
            target_robot_qpos
            .unsqueeze(0)
            .expand(num_envs, -1)
        )

        target_object_pos_batch = (
            target_object_pos
            .unsqueeze(0)
            .expand(num_envs, -1)
        )

        target_object_quat_batch = (
            target_object_quat
            .unsqueeze(0)
            .expand(num_envs, -1)
        )

        state = _make_state(
            robot_qpos,
            object_pos,
            object_quat,
            target_robot_qpos_batch,
            target_object_pos_batch,
            target_object_quat_batch,
        )

        # ========================================================
        # ACTOR
        # ========================================================

        policy_dist = actor(
            robot_qpos,
            object_pos,
            object_quat,
            target_robot_qpos_batch,
            target_object_pos_batch,
            target_object_quat_batch,
        )

        raw_action = policy_dist.sample()

        old_log_prob = (
            policy_dist
            .log_prob(raw_action)
            .sum(dim=-1)
        )

        # ========================================================
        # ACTION = DELTA Q
        # ========================================================

        delta_q = (
            action_scale * raw_action
        )

        target_qpos = robot_qpos + delta_q

        # Joint limits.
        target_qpos = torch.maximum(
            target_qpos,
            robot_q_min,
        )

        target_qpos = torch.minimum(
            target_qpos,
            robot_q_max,
        )

        # IMPORTANT:
        # control_dofs_position creates a physical PD command.
        # set_dofs_position would teleport the robot and destroy
        # the intended manipulation dynamics.
        robot.control_dofs_position(
            target_qpos,
            dofs_idx_local=motors_dof_idx,
        )

        # ========================================================
        # VIRTUAL OBJECT CONTROLLER (VOC)
        #
        # The object is physically pulled toward the expected
        # demonstration pose. This is NOT a teleport.
        #
        # VOC is strongest early in training and is progressively
        # reduced as the policy produces better contact.
        # ========================================================

        object_lin_vel = obj.get_links_vel(
            links_idx_local=[0]
        )[:, 0, :]

        target_object_lin_vel = (
            demo_object_positions[target_frame]
            - demo_object_positions[frame]
        ) / (1.0 / 30.0)

        target_object_lin_vel = (
            target_object_lin_vel
            .to(device=device, dtype=dtype)
            .unsqueeze(0)
            .expand(num_envs, -1)
        )

        _apply_virtual_object_controller(
            obj=obj,
            scene=scene,
            current_pos=object_pos,
            current_quat=object_quat,
            current_lin_vel=object_lin_vel,
            target_pos=target_object_pos_batch,
            target_quat=target_object_quat_batch,
            target_lin_vel=target_object_lin_vel,
            voc_strength=voc_strength,
        )

        # ========================================================
        # PHYSICS
        # ========================================================

        scene.step()

        # ========================================================
        # POST-PHYSICS OBSERVATION
        # ========================================================

        current_keypoints = robot.get_links_pos()

        current_contacts = robot.get_contacts(
            with_entity=obj,
            # is_padded=True,
        )

        object_pos_next = obj.get_pos()
        object_quat_next = obj.get_quat()

        # ========================================================
        # REWARD
        # ========================================================

        reward_terms = reward_module.compute_reward_terms(
            current_keypoints=current_keypoints,
            current_contacts=current_contacts,
            object_pos=object_pos_next,
            object_quat=object_quat_next,
            delta_q=delta_q,
            frame_id=frame,
            target_frame=target_frame,
        )

        reward = reward_terms["total"]
        contact_reward = reward_terms["contact"]

        total_reward += reward

        # --------------------------------------------------------
        # VOC curriculum update
        #
        # Contact reward is approximately [0, 1]. We only decay
        # after contact quality passes a useful threshold, so
        # accidental/no-contact states do not remove assistance.
        # --------------------------------------------------------

        voc_contact_ema = (
            0.9 * voc_contact_ema
            + 0.1 * contact_reward.detach()
        )

        contact_progress = (
            (voc_contact_ema - voc_contact_threshold)
            / max(1.0 - voc_contact_threshold, 1e-6)
        ).clamp(0.0, 1.0)

        voc_strength = torch.maximum(
            torch.full_like(
                voc_strength,
                voc_min_strength,
            ),
            voc_strength
            * torch.exp(
                -voc_decay_rate * contact_progress
            ),
        )

        # ========================================================
        # CRITIC
        #
        # Value belongs to the state BEFORE the action.
        # ========================================================

        value = critic(
            robot_qpos,
            object_pos,
            object_quat,
            target_robot_qpos_batch,
            target_object_pos_batch,
            target_object_quat_batch,
        )

        # ========================================================
        # STORE
        # ========================================================

        states.append(state.detach())
        actions.append(raw_action.detach())
        rewards.append(reward.detach())
        log_probs.append(old_log_prob.detach())
        values.append(value.detach())

    return {
        "states": torch.stack(states),        # [T,N,S]
        "actions": torch.stack(actions),      # [T,N,D]
        "rewards": torch.stack(rewards),      # [T,N]
        "old_log_probs": torch.stack(log_probs),  # [T,N]
        "values": torch.stack(values),        # [T,N]
        "total_reward": total_reward,         # [N]
        "voc_strength": voc_strength.detach(),
        "voc_contact_ema": voc_contact_ema.detach(),
    }


def compute_gae(
    rewards,
    values,
    gamma=0.99,
    gae_lambda=0.95,
):
    """
    Batched GAE.

    rewards: [T,N]
    values : [T,N]
    """

    T = rewards.shape[0]

    advantages = torch.zeros_like(
        rewards
    )

    gae = torch.zeros_like(
        rewards[0]
    )

    for t in reversed(range(T)):

        if t == T - 1:
            next_value = torch.zeros_like(
                values[t]
            )
        else:
            next_value = values[t + 1]

        delta = (
            rewards[t]
            + gamma * next_value
            - values[t]
        )

        gae = (
            delta
            + gamma
            * gae_lambda
            * gae
        )

        advantages[t] = gae

    returns = advantages + values

    advantages = (
        advantages - advantages.mean()
    ) / (
        advantages.std(unbiased=False) + 1e-8
    )

    return advantages, returns


def ppo_update(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    rollout,
    advantages,
    returns,
    ppo_epochs=10,
    minibatch_size=4096,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.005,
):
    states = rollout["states"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]

    T, N, state_dim = states.shape
    action_dim = actions.shape[-1]

    states = states.reshape(
        T * N,
        state_dim,
    )

    actions = actions.reshape(
        T * N,
        action_dim,
    )

    old_log_probs = old_log_probs.reshape(
        T * N
    )

    advantages = advantages.reshape(
        T * N
    ).detach()

    returns = returns.reshape(
        T * N
    ).detach()

    # Actor stores robot_dof.
    D = actor.robot_dof

    robot_qpos = states[:, :D]
    object_pos = states[:, D:D + 3]
    object_quat = states[:, D + 3:D + 7]

    target_robot_qpos = states[:, D + 7:D + 7 + D]
    target_object_pos = states[:, D + 7 + D:D + 10 + D]
    target_object_quat = states[:, D + 10 + D:D + 14 + D]

    total_samples = states.shape[0]

    last_actor_loss = 0.0
    last_critic_loss = 0.0
    last_ratio = 1.0

    for _ in range(ppo_epochs):

        permutation = torch.randperm(
            total_samples,
            device=states.device,
        )

        for start in range(
            0,
            total_samples,
            minibatch_size,
        ):
            idx = permutation[
                start:start + minibatch_size
            ]

            dist = actor(
                robot_qpos[idx],
                object_pos[idx],
                object_quat[idx],
                target_robot_qpos[idx],
                target_object_pos[idx],
                target_object_quat[idx],
            )

            new_log_probs = (
                dist.log_prob(
                    actions[idx]
                ).sum(dim=-1)
            )

            entropy = (
                dist.entropy()
                .sum(dim=-1)
                .mean()
            )

            ratio = torch.exp(
                new_log_probs
                - old_log_probs[idx]
            )

            unclipped = (
                ratio * advantages[idx]
            )

            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - clip_eps,
                    1.0 + clip_eps,
                )
                * advantages[idx]
            )

            policy_loss = -torch.min(
                unclipped,
                clipped,
            ).mean()

            actor_loss = (
                policy_loss
                - entropy_coef * entropy
            )

            actor_optimizer.zero_grad(
                set_to_none=True
            )

            actor_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                0.5,
            )

            actor_optimizer.step()

            # ----------------------------------------------------
            # Critic update
            # ----------------------------------------------------

            predicted_value = critic(
                robot_qpos[idx],
                object_pos[idx],
                object_quat[idx],
                target_robot_qpos[idx],
                target_object_pos[idx],
                target_object_quat[idx],
            )

            critic_loss = (
                returns[idx] - predicted_value
            ).pow(2).mean()

            critic_loss = (
                value_coef * critic_loss
            )

            critic_optimizer.zero_grad(
                set_to_none=True
            )

            critic_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                0.5,
            )

            critic_optimizer.step()

            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_ratio = ratio.mean().item()

    return (
        last_actor_loss,
        last_critic_loss,
        last_ratio,
    )


def train_parallel_episode(
    reward_module,
    actor,
    critic,
    scene,
    robot,
    obj,
    timeline_len,
    motors_dof_idx,
    demo_robot_qpos,
    demo_object_positions,
    demo_object_quaternions,
    num_episodes=100,
    action_scale=0.1,
):
    device = robot.get_qpos().device

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=3e-4,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=1e-3,
    )

    robot_q_min = robot.get_dofs_limit(
        dofs_idx_local=motors_dof_idx
    )[0].to(device)

    robot_q_max = robot.get_dofs_limit(
        dofs_idx_local=motors_dof_idx
    )[1].to(device)

    # Fallback if Genesis returns limits as a tuple/list.
    if robot_q_min.ndim == 0:
        robot_q_min = torch.as_tensor(
            robot_q_min,
            device=device,
        )

    if robot_q_max.ndim == 0:
        robot_q_max = torch.as_tensor(
            robot_q_max,
            device=device,
        )

    # Persistent curriculum state. It is intentionally NOT reset
    # every PPO episode: the controller should hand control from VOC
    # to the policy over the course of training.
    voc_strength = torch.ones(
        robot.get_qpos().shape[0],
        device=device,
        dtype=robot.get_qpos().dtype,
    )

    voc_contact_ema = torch.zeros_like(voc_strength)

    best_reward = -float("inf")

    for episode in range(num_episodes):

        rollout = train_one_episode(
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
            robot_q_min=robot_q_min,
            robot_q_max=robot_q_max,
            action_scale=action_scale,
            voc_strength=voc_strength,
            voc_contact_ema=voc_contact_ema,
        )

        voc_strength = rollout["voc_strength"]
        voc_contact_ema = rollout["voc_contact_ema"]

        advantages, returns = compute_gae(
            rollout["rewards"],
            rollout["values"],
        )

        actor_loss, critic_loss, ratio = ppo_update(
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            rollout=rollout,
            advantages=advantages,
            returns=returns,
        )

        episode_rewards = rollout["total_reward"]

        mean_reward = episode_rewards.mean()
        best_episode_reward, best_env = (
            episode_rewards.max(dim=0)
        )

        if best_episode_reward.item() > best_reward:

            best_reward = best_episode_reward.item()

            # Save the actual best candidate trajectory.
            best_actions = rollout["actions"][
                :, best_env
            ].detach().cpu()

            torch.save(
                {
                    "episode": episode,
                    "best_env": best_env.item(),
                    "reward": best_reward,
                    "best_actions": best_actions,
                    "voc_strength": voc_strength.detach().cpu(),
                    "voc_config": {
                        "initial_strength": 1.0,
                        "decay_rate": 0.02,
                        "contact_threshold": 0.65,
                        "min_strength": 0.0,
                    },
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "actor_optimizer_state_dict":
                        actor_optimizer.state_dict(),
                    "critic_optimizer_state_dict":
                        critic_optimizer.state_dict(),
                },
                CHECKPOINT_DIR / f"best_ppo_{datetime.now()}.pt",
            )

        print(
            f"Episode {episode:04d} | "
            f"Mean {mean_reward.item():.4f} | "
            f"Best {best_episode_reward.item():.4f} "
            f"(env {best_env.item()}) | "
            f"Actor {actor_loss:.5f} | "
            f"Critic {critic_loss:.5f} | "
            f"Ratio {ratio:.4f} | "
            f"VOC mean {voc_strength.mean().item():.4f}"
        )
