from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class Actor(nn.Module):
    """
    PPO actor for trajectory-conditioned position control.

    Observation:
        current robot qpos
        current object position + quaternion
        desired robot qpos
        desired object position + quaternion

    Action:
        normalized delta-q command. The environment converts it to:
            target_qpos = current_qpos + action_scale * action
    """

    def __init__(self, robot_dof: int):
        super().__init__()

        self.robot_dof = robot_dof

        # current robot       : robot_dof
        # current object      : 7
        # desired robot       : robot_dof
        # desired object      : 7
        state_dim = 2 * robot_dof + 14
        action_dim = robot_dof

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
        )

        # Initial std ~= 0.135 in normalized action space.
        self.log_std = nn.Parameter(
            torch.full((action_dim,), -2.0)
        )

    def forward(
        self,
        robot_qpos: torch.Tensor,
        object_pos: torch.Tensor,
        object_quat: torch.Tensor,
        target_robot_qpos: torch.Tensor,
        target_object_pos: torch.Tensor,
        target_object_quat: torch.Tensor,
    ):
        state = torch.cat(
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

        mean = self.network(state)
        std = torch.exp(self.log_std)

        return Normal(mean, std)

    def sample(
        self,
        robot_qpos,
        object_pos,
        object_quat,
        target_robot_qpos,
        target_object_pos,
        target_object_quat,
    ):
        dist = self(
            robot_qpos,
            object_pos,
            object_quat,
            target_robot_qpos,
            target_object_pos,
            target_object_quat,
        )

        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        return raw_action, log_prob, dist


class Critic(nn.Module):

    def __init__(self, robot_dof: int):
        super().__init__()

        self.robot_dof = robot_dof

        state_dim = 2 * robot_dof + 14

        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        robot_qpos,
        object_pos,
        object_quat,
        target_robot_qpos,
        target_object_pos,
        target_object_quat,
    ):
        state = torch.cat(
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

        return self.network(state).squeeze(-1)
