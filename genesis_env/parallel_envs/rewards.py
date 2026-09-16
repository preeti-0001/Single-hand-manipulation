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
        demo_ideal_grasps, 
        beta_imitation=0.1,
        beta_contact=10.0,
        beta_position=0.1,
        beta_rotation=0.5,
        beta_bc=0.2,
        beta_grasp_position=20.0,
        beta_grasp_qpos=1.0,
        lambda_grasp=3.0,
        lambda_task=0.5,
        lambda_imitation=0.5,
        lambda_contact=3.0,
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
        self.beta_grasp_position = beta_grasp_position
        self.beta_grasp_qpos = beta_grasp_qpos
        self.lambda_grasp = lambda_grasp
        self.demo_ideal_grasps = demo_ideal_grasps

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

    def compute_grasp_reward(
        self,
        current_keypoints,
        current_contacts,
        current_qpos,
    ):
        """
        Reward the robot for approaching the demonstrated ideal grasp.
    
        Returns:
            [num_envs]
        """
    
        device = current_qpos.device
        dtype = current_qpos.dtype
    
        # --------------------------------------------------------
        # Ideal grasp
        # --------------------------------------------------------
    
        grasp = self.demo_ideal_grasps[0]
    
        target_position = self._tensor(
            grasp["position"],
            device,
            dtype,
        )
    
        target_qpos = self._tensor(
            grasp["qpos"],
            device,
            dtype,
        )
    
        target_contact_links = self._tensor(
            grasp["contact_links"],
            device,
        ).bool()
    
        # --------------------------------------------------------
        # 1. GRASP POSITION
        # --------------------------------------------------------
    
        # Use current contact centroid when contacts exist.
        positions = current_contacts["position"]
        valid = current_contacts["valid_mask"]
    
        has_contact = valid.any(dim=-1)
    
        current_centroid = torch.zeros(
            positions.shape[0],
            3,
            device=device,
            dtype=dtype,
        )
    
        for i in range(positions.shape[0]):
    
            if has_contact[i]:
    
                current_centroid[i] = (
                    positions[i][valid[i]].mean(dim=0)
                )
    
        position_error = (
            current_centroid - target_position.unsqueeze(0)
        ).pow(2).sum(dim=-1).sqrt()
    
        position_reward = torch.exp(
            -self.beta_grasp_position * position_error
        )
    
        # No contact -> no grasp-position reward
        position_reward = torch.where(
            has_contact,
            position_reward,
            torch.zeros_like(position_reward),
        )
    
        # --------------------------------------------------------
        # 2. GRASP QPOS
        # --------------------------------------------------------
    
        qpos_error = (
            current_qpos
            - target_qpos.unsqueeze(0)
        ).pow(2).mean(dim=-1)
    
        qpos_reward = torch.exp(
            -self.beta_grasp_qpos * qpos_error
        )
    
        # --------------------------------------------------------
        # 3. CONTACT-LINK MATCHING
        # --------------------------------------------------------
    
        # Current contact information should provide link IDs
        # if available.
        current_contact_links = current_contacts.get(
            "link_mask",
            None,
        )
    
        if current_contact_links is None:
    
            contact_link_reward = torch.zeros(
                current_qpos.shape[0],
                device=device,
                dtype=dtype,
            )
    
        else:
    
            current_contact_links = (
                current_contact_links.bool()
            )
    
            target = target_contact_links.unsqueeze(0)
    
            intersection = (
                current_contact_links & target
            ).sum(dim=-1).float()
    
            target_count = (
                target_contact_links.sum()
                .clamp_min(1)
            )
    
            contact_link_reward = (
                intersection / target_count
            )
    
        # --------------------------------------------------------
        # FINAL GRASP REWARD
        # --------------------------------------------------------
    
        grasp_reward = (
            0.4 * position_reward
            + 0.3 * qpos_reward
            + 0.3 * contact_link_reward
        )
    
        return grasp_reward

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

    def compute_reward_terms(
        self,
        current_keypoints,
        current_qpos,
        current_contacts,
        object_pos,
        object_quat,
        delta_q,
        frame_id,
        target_frame,
    ):
        """Return individual reward terms before weighting."""
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

        bc = self.compute_behaviour_cloning_reward(
            delta_q,
            frame_id,
        )

        grasp = self.compute_grasp_reward(
            current_qpos=current_qpos,
            current_keypoints=current_keypoints,
            current_contacts=current_contacts,
        )

        total = (
            self.lambda_task * task
            + self.lambda_imitation * imitation
            + self.lambda_contact * contact
            + self.lambda_bc * bc
            + self.lambda_grasp * grasp
        )

        return {
            "task": task,
            "imitation": imitation,
            "contact": contact,
            "bc": bc,
            "grasp": grasp,
            "total": total,
        }

    def compute_total_reward(
        self,
        current_keypoints,
        current_qpos,
        current_contacts,
        object_pos,
        object_quat,
        delta_q,
        frame_id,
        target_frame,
    ):
        return self.compute_reward_terms(
            current_keypoints=current_keypoints,
            current_qpos=current_qpos,
            current_contacts=current_contacts,
            object_pos=object_pos,
            object_quat=object_quat,
            delta_q=delta_q,
            frame_id=frame_id,
            target_frame=target_frame,
        )["total"]
