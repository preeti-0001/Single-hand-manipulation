import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian PPO policy with tanh-bounded actions in [-1, 1]."""

    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, obs):
        x = self.net(obs)
        mean = self.mean(x)
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std

    def distribution(self, obs):
        mean, std = self(obs)
        return Normal(mean, std)

    def sample(self, obs):
        dist = self.distribution(obs)
        raw = dist.rsample()
        action = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1)

    def evaluate_actions(self, obs, actions):
        actions = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        raw = torch.atanh(actions)
        dist = self.distribution(obs)
        log_prob = dist.log_prob(raw) - torch.log(1.0 - actions.pow(2) + 1e-6)
        return log_prob.sum(-1), dist.entropy().sum(-1)
