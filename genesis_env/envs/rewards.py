import torch
from src.utils.reward_utils import position_distance, rotation_distance
from scipy.spatial.transform import Rotation


class RewardModule:
    def __init__(
        self,
        demo_contact_tensor,
        demo_object_trajectories,
        demo_object_quaternions,
        demo_robot_trajectories,
        demo_contact_validity,
        hand_dof,
        arm_dof,
        timeline_len,
        demo_actions,
        beta_imitation=0.1,
        beta_contact=0.3,
        beta_position=1.0,
        beta_rotation=20.0,
        beta_angle=5.0,
        beta_bc=2.0,
        lambda_task=0.5,
        lambda_imitation=0.1,
        lambda_contact=0.3,
        lambda_bc=0.1,
        contact_dmax=0.05,
    ):
        self.beta_imitation = beta_imitation
        self.beta_contact = beta_contact
        self.beta_position = beta_position
        self.beta_rotation = beta_rotation
        self.beta_angle = beta_angle
        self.beta_bc = beta_bc
        self.lambda_task = lambda_task
        self.lambda_imitation = lambda_imitation
        self.lambda_contact = lambda_contact
        self.lambda_bc = lambda_bc
        self.contact_dmax = contact_dmax
        self.demo_contact_tensor = demo_contact_tensor
        self.demo_object_trajectories = demo_object_trajectories
        self.demo_object_quaternions = demo_object_quaternions
        self.demo_robot_trajectories = demo_robot_trajectories
        self.demo_contact_validity = demo_contact_validity
        self.demo_actions = demo_actions
        self.hand_dof = hand_dof
        self.arm_dof = arm_dof
        self.timeline_len = timeline_len

    def compute_motion_imitation_reward(self, current_keypoints, frame_id):
        demo_keypoints = self.demo_robot_trajectories[frame_id]
        pred_keypoints = current_keypoints
        demo = torch.stack(demo_keypoints)  # [23, 3]
        pred = torch.stack(pred_keypoints)  # [23, 3]

        # Squared Euclidean distance for each keypoint
        dist_sq = torch.sum((pred - demo) ** 2, dim=1)  # [23]

        r_i = torch.exp(-self.beta_imitation * dist_sq)  # [23]

        # Overall reward
        R = r_i.mean()
        return R

    def compute_contact_reward(self, current_contacts, frame_id):

        # ==================================================
        # Genesis actual contact positions
        # ==================================================

        current_positions = current_contacts["position"]

        # ==================================================
        # Demo contact data
        # ==================================================

        demo_positions = torch.as_tensor(
            self.demo_contact_tensor[frame_id],
            dtype=current_positions.dtype,
            device=current_positions.device,
        ).squeeze(
            0
        )  # (23, 3)

        demo_validity = torch.as_tensor(
            self.demo_contact_validity[frame_id],
            dtype=torch.bool,
            device=current_positions.device,
        ).squeeze(
            0
        )  # (23,)

        # Only contact points that actually exist
        demo_positions = demo_positions[demo_validity]

        # No demonstrated contacts
        if demo_positions.shape[0] == 0:

            # If there are no demo contacts and no actual contacts,
            # that's correct.
            if current_positions.shape[0] == 0:
                return torch.ones(
                    (),
                    device=current_positions.device,
                    dtype=current_positions.dtype,
                )

            # Demo says no contact, but robot has contact.
            return torch.zeros(
                (),
                device=current_positions.device,
                dtype=current_positions.dtype,
            )

        # ==================================================
        # Demo has contacts, but robot currently has none
        # ==================================================

        if current_positions.shape[0] == 0:

            return torch.exp(
                torch.tensor(
                    -self.beta_contact * self.contact_dmax,
                    device=current_positions.device,
                    dtype=current_positions.dtype,
                )
            )

        # ==================================================
        # Distance between demo and actual contacts
        # ==================================================

        distances = torch.cdist(
            demo_positions,
            current_positions,
            p=2.0,
        )

        # For each DEMO contact, find closest actual contact
        min_distances = distances.min(dim=1).values

        # Prevent excessively large distances
        min_distances = torch.clamp(
            min_distances,
            max=self.contact_dmax,
        )

        # ==================================================
        # Contact reward
        # ==================================================

        reward = torch.exp(-self.beta_contact * min_distances)

        return reward.mean()

    def compute_task_reward(self, object_pos, object_quat, frame_id):
        demo_object_pos = self.demo_object_trajectories[frame_id]
        demo_object_quat = self.demo_object_quaternions[frame_id]
        dpos = position_distance(object_pos, demo_object_pos)
        drot = rotation_distance(object_quat, demo_object_quat)
        object_ang = 2.0 * torch.acos(
            torch.clamp(torch.abs(object_quat[..., 0]), max=1.0)
        )

        demo_object_ang = 2.0 * torch.acos(
            torch.clamp(torch.abs(demo_object_quat[..., 0]), max=1.0)
        )

        dang = torch.abs(object_ang - demo_object_ang)
        return (
            torch.exp(-self.beta_position * dpos)
            * torch.exp(-self.beta_rotation * drot)
            * torch.exp(-self.beta_angle * dang)
        )

    def compute_behaviour_cloning_reward(self, policy_action, frame_id):
        dist_sq = (policy_action - self.demo_actions[frame_id]) ** 2

        r_i = torch.exp(-self.beta_bc * dist_sq)

        r_bc = r_i.mean()
        return r_bc

    def compute_total_reward(
        self,
        current_keypoints,
        current_contacts,
        object_pos,
        object_quat,
        policy_action,
        frame_id,
    ):
        z = torch.zeros_like(
            task_reward := self.compute_task_reward(object_pos, object_quat, frame_id)
        )
        contact_reward = (
            z
            if self.compute_contact_reward(current_contacts, frame_id) is None
            else self.compute_contact_reward(current_contacts, frame_id)
        )
        imitation_reward = (
            z
            if self.compute_motion_imitation_reward(current_keypoints, frame_id) is None
            else self.compute_motion_imitation_reward(current_keypoints, frame_id)
        )
        behaviour_cloning_reward = (
            z
            if self.compute_behaviour_cloning_reward(policy_action, frame_id) is None
            else self.compute_behaviour_cloning_reward(policy_action, frame_id)
        )
        return (
            self.lambda_task * task_reward
            + self.lambda_imitation * imitation_reward
            + self.lambda_contact * contact_reward
            + self.lambda_bc * behaviour_cloning_reward
        )
