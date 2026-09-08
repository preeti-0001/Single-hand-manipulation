import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, robot_dof):
        super().__init__()

        state_dim = robot_dof + 7 # Robot qpos + Object positions + quaternion
        action_dim = robot_dof

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, action_dim),
        )

        # Learnable exploration noise
        self.log_std = nn.Parameter(
            torch.zeros(action_dim)
        )

    def forward(self, robot_qpos, object_pos, object_quat):
        state = torch.cat(
            [robot_qpos, object_pos, object_quat],
            dim=-1
        )

        mean = self.network(state)

        std = torch.exp(self.log_std)

        return Normal(mean, std)
    


class Critic(nn.Module):
    def __init__(self, robot_dof):
        super().__init__()

        state_dim = robot_dof + 7

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),

            nn.Linear(256, 256),
            nn.Tanh(),

            nn.Linear(256, 1),
        )

    def forward(self, robot_qpos, object_pos, object_quat):
        state = torch.cat(
            [robot_qpos, object_pos, object_quat],
            dim=-1
        )

        return self.network(state).squeeze(-1)

