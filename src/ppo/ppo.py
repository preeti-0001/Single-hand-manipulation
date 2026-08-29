from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn

from .actor import Actor
from .critic import Critic


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    value_learning_rate: float = 1e-3
    update_epochs: int = 10
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.obs, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

    def add(self, obs, action, log_prob, reward, value, done):
        self.obs.append(obs.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        self.rewards.append(torch.as_tensor(reward, device=obs.device).detach())
        self.values.append(value.detach())
        self.dones.append(torch.as_tensor(done, device=obs.device).detach())

    def tensors(self):
        return tuple([
            torch.stack(self.obs),
            torch.stack(self.actions),
            torch.stack(self.log_probs).reshape(-1),
            torch.stack(self.rewards).reshape(-1),
            torch.stack(self.values).reshape(-1),
            torch.stack(self.dones).reshape(-1),
        ])


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    rewards, values = rewards.reshape(-1), values.reshape(-1)
    dones = dones.reshape(-1).float()
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(len(rewards))):
        next_value = last_value if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[t] = gae

    return advantages, advantages + values


class PPO:
    def __init__(self, actor: Actor, critic: Critic, config=None):
        self.actor, self.critic = actor, critic
        self.cfg = config or PPOConfig()
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=self.cfg.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self.cfg.value_learning_rate)

    @torch.no_grad()
    def act(self, obs):
        single = obs.ndim == 1
        if single:
            obs = obs.unsqueeze(0)
        action, log_prob = self.actor.sample(obs)
        value = self.critic(obs)
        if single:
            return action[0], log_prob[0], value[0]
        return action, log_prob, value

    def update(self, buffer, last_value):
        obs, actions, old_lp, rewards, values, dones = buffer.tensors()
        adv, returns = compute_gae(rewards, values, dones, last_value.detach(), self.cfg.gamma, self.cfg.gae_lambda)
        if self.cfg.normalize_advantage:
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        n, mb = len(obs), min(self.cfg.minibatch_size, len(obs))
        stats = {k: 0.0 for k in ["actor_loss", "critic_loss", "entropy", "approx_kl", "clip_fraction"]}
        count = 0

        for _ in range(self.cfg.update_epochs):
            perm = torch.randperm(n, device=obs.device)
            for start in range(0, n, mb):
                idx = perm[start:start + mb]
                new_lp, entropy = self.actor.evaluate_actions(obs[idx], actions[idx])
                new_values = self.critic(obs[idx])
                ratio = torch.exp(new_lp - old_lp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv[idx]
                actor_loss = -torch.min(s1, s2).mean()
                critic_loss = 0.5 * (returns[idx] - new_values).pow(2).mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                (actor_loss - self.cfg.entropy_coef * entropy.mean()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad(set_to_none=True)
                (self.cfg.value_coef * critic_loss).backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.critic_optimizer.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy.mean().item()
                stats["approx_kl"] += (old_lp[idx] - new_lp).mean().item()
                stats["clip_fraction"] += (torch.abs(ratio - 1) > self.cfg.clip_eps).float().mean().item()
                count += 1

        buffer.clear()
        for k in stats:
            stats[k] /= max(count, 1)
        stats["mean_reward"] = rewards.mean().item()
        return stats


def train(env, actor, critic, num_iterations=1000, max_steps=None, config=None):
    ppo = PPO(actor, critic, config)
    buffer = RolloutBuffer()
    horizon = max_steps or getattr(env, "demo_length", 1000)

    for iteration in range(num_iterations):
        obs = env.reset()
        done, episode_reward, steps = False, 0.0, 0
        while not done and steps < horizon:
            action, log_prob, value = ppo.act(obs)
            next_obs, reward, done, info = env.step(action)
            buffer.add(obs, action, log_prob, reward, value, done)
            obs = next_obs
            episode_reward += float(torch.as_tensor(reward).mean().item())
            steps += 1

        with torch.no_grad():
            last_value = torch.zeros((), device=obs.device, dtype=obs.dtype) if done else critic(obs)
        stats = ppo.update(buffer, last_value)
        print(f"Iteration {iteration:04d} | steps={steps:4d} | reward={episode_reward:9.4f} | actor={stats['actor_loss']:.4f} | critic={stats['critic_loss']:.4f} | entropy={stats['entropy']:.4f} | KL={stats['approx_kl']:.6f}")

    return ppo
