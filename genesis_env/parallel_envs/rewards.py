from __future__ import annotations

import torch
from src.utils.reward_utils import position_distance, rotation_distance


class RewardModule:
    """
    All reward functions are batch-aware.

    Every reward returned by this class has shape [num_envs].
    """

    def __init__(
        self,
        demo_contact_tensor,
        demo_contact_validity,
        demo_robot_keypoints,
        demo_robot_qpos,
        demo_object_trajectories,
        demo_object_quaternions,
        beta_imitation=10.0,
        beta_contact=10.0,
        beta_position=10.0,
        beta_rotation=5.0,
        beta_bc=2.0,
        lambda_task=1.0,
        lambda_imitation=1.0,
        lambda_contact=0.2,
        lambda_bc=0.1,
        contact_dmax=0.05,
    ):
        self.demo_contact_tensor = demo_contact_tensor
        self.demo_contact_validity = demo_contact_validity

        self.demo_robot_keypoints = demo_robot_keypoints
        self.demo_robot_qpos = demo_robot_qpos

        self.demo_object_trajectories = demo_object_trajectories
        self.demo_object_quaternions = demo_object_quaternions

        self.beta_imitation = beta_imitation
        self.beta_contact = beta_contact
        self.beta_position = beta_position
        self.beta_rotation = beta_rotation
        self.beta_bc = beta_bc

        self.lambda_task = lambda_task
        self.lambda_imitation = lambda_imitation
        self.lambda_contact = lambda_contact
        self.lambda_bc = lambda_bc

        self.contact_dmax = contact_dmax

    @staticmethod
    def _tensor(x, device, dtype=None):
        t = torch.as_tensor(x, device=device)
        if dtype is not None:
            t = t.to(dtype)
        return t

    def compute_motion_imitation_reward(
        self,
        current_keypoints,
        target_frame,
    ):
        # current_keypoints: [N, L, 3]
        demo = self._tensor(
            self.demo_robot_keypoints[target_frame],
            current_keypoints.device,
            current_keypoints.dtype,
        )  # [L, 3]

        dist_sq = (
            current_keypoints - demo.unsqueeze(0)
        ).pow(2).sum(dim=-1)

        # Mean link error -> reward in (0,1].
        return torch.exp(
            -self.beta_imitation * dist_sq
        ).mean(dim=-1)

    def compute_task_reward(
        self,
        object_pos,
        object_quat,
        target_frame,
    ):
        demo_pos = self._tensor(
            self.demo_object_trajectories[target_frame],
            object_pos.device,
            object_pos.dtype,
        )

        demo_quat = self._tensor(
            self.demo_object_quaternions[target_frame],
            object_quat.device,
            object_quat.dtype,
        )

        # Normalize quaternions before using them.
        object_quat = object_quat / (
            object_quat.norm(dim=-1, keepdim=True) + 1e-8
        )
        demo_quat = demo_quat / (
            demo_quat.norm(dim=-1, keepdim=True) + 1e-8
        )

        dpos = position_distance(
            object_pos,
            demo_pos,
        )

        drot = rotation_distance(
            object_quat,
            demo_quat,
        )

        return (
            torch.exp(-self.beta_position * dpos)
            * torch.exp(-self.beta_rotation * drot)
        )

    def compute_behaviour_cloning_reward(
        self,
        delta_q,
        frame_id,
    ):
        if frame_id >= len(self.demo_robot_qpos) - 1:
            return torch.ones(
                delta_q.shape[0],
                device=delta_q.device,
                dtype=delta_q.dtype,
            )

        q0 = self._tensor(
            self.demo_robot_qpos[frame_id],
            delta_q.device,
            delta_q.dtype,
        )
        q1 = self._tensor(
            self.demo_robot_qpos[frame_id + 1],
            delta_q.device,
            delta_q.dtype,
        )

        demo_delta = q1 - q0

        error = (delta_q - demo_delta.unsqueeze(0)).pow(2)

        return torch.exp(
            -self.beta_bc * error
        ).mean(dim=-1)

    def compute_contact_reward(
        self,
        current_contacts,
        frame_id,
    ):
        """
        Genesis parallel contact output:
            position   : [N, C, 3]
            valid_mask : [N, C]

        Demo contacts:
            [demo_contacts, 3]
            validity:
            [demo_contacts]
        """

        positions = current_contacts["position"]
        valid = current_contacts["valid_mask"]

        device = positions.device
        dtype = positions.dtype

        demo_positions = self._tensor(
            self.demo_contact_tensor[frame_id],
            device,
            dtype,
        ).squeeze(0)

        demo_valid = self._tensor(
            self.demo_contact_validity[frame_id],
            device,
        ).squeeze(0).bool()

        demo_positions = demo_positions[demo_valid]

        num_envs = positions.shape[0]

        # No demonstrated contacts.
        if demo_positions.numel() == 0:
            has_contact = valid.any(dim=-1)

            return torch.where(
                has_contact,
                torch.zeros(
                    num_envs,
                    device=device,
                    dtype=dtype,
                ),
                torch.ones(
                    num_envs,
                    device=device,
                    dtype=dtype,
                ),
            )

        # [N, D, C, 3]
        distances = torch.cdist(
            demo_positions.unsqueeze(0),
            positions,
        )

        # Invalid current contacts should never be selected.
        distances = distances.masked_fill(
            ~valid.unsqueeze(1),
            float("inf"),
        )

        # [N, D]
        min_distances = distances.min(dim=-1).values

        has_current = valid.any(dim=-1)

        min_distances = torch.clamp(
            min_distances,
            max=self.contact_dmax,
        )

        reward = torch.exp(
            -self.beta_contact * min_distances
        ).mean(dim=-1)

        no_contact_reward = torch.exp(
            torch.tensor(
                -self.beta_contact * self.contact_dmax,
                device=device,
                dtype=dtype,
            )
        )

        return torch.where(
            has_current,
            reward,
            no_contact_reward.expand(num_envs),
        )

    def compute_total_reward(
        self,
        current_keypoints,
        current_contacts,
        object_pos,
        object_quat,
        delta_q,
        frame_id,
        target_frame,
    ):
        task = self.compute_task_reward(
            object_pos,
            object_quat,
            target_frame,
        )

        imitation = self.compute_motion_imitation_reward(
            current_keypoints,
            target_frame,
        )

        contact = self.compute_contact_reward(
            current_contacts,
            target_frame,
        )
        contact_quality = contact.clone()

        bc = self.compute_behaviour_cloning_reward(
            delta_q,
            frame_id,
        )

        return (
            self.lambda_task * task
            + self.lambda_imitation * imitation
            + self.lambda_contact * contact
            + self.lambda_bc * bc
        ), contact_quality
