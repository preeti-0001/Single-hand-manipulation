import torch
from src.utils.reward_utils import position_distance, rotation_distance


class RewardModule:
    def __init__(self, beta_imitation=1.0, beta_contact=5.0, beta_position=5.0,
                 beta_rotation=2.0, beta_angle=1.0, beta_bc=2.0,
                 lambda_task=1.0, lambda_imitation=0.5, lambda_contact=1.0,
                 lambda_bc=0.25, contact_dmax=0.05):
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

    def compute_motion_imitation_reward(self, current_keypoints, demo_keypoints):
        d = torch.linalg.norm(current_keypoints - demo_keypoints, dim=-1)
        return torch.exp(-self.beta_imitation * d).mean(dim=-1)

    def compute_contact_reward(self, current_contacts, demo_contacts, current_validity, demo_validity):
        cv, dv = current_validity.bool(), demo_validity.bool()
        d = torch.linalg.norm(current_contacts - demo_contacts, dim=-1)
        relevant = cv | dv
        D = torch.where(cv & dv, d, torch.where(cv ^ dv, torch.full_like(d, self.contact_dmax), torch.zeros_like(d)))
        r = torch.exp(-self.beta_contact * D)
        return r[relevant].mean() if relevant.any() else torch.zeros((), device=r.device, dtype=r.dtype)

    def compute_task_reward(self, object_pos, target_pos, object_rot, target_rot, object_ang, target_ang):
        dpos = position_distance(object_pos, target_pos)
        drot = rotation_distance(object_rot, target_rot)
        dang = torch.linalg.norm(object_ang - target_ang, dim=-1)
        return torch.exp(-self.beta_position*dpos) * torch.exp(-self.beta_rotation*drot) * torch.exp(-self.beta_angle*dang)

    def compute_behaviour_cloning_reward(self, policy_action, demo_action):
        return torch.exp(-self.beta_bc * torch.abs(policy_action - demo_action)).mean(dim=-1)

    def compute_total_reward(self, task_reward, contact_reward, imitation_reward=None, behaviour_cloning_reward=None):
        z = torch.zeros_like(task_reward)
        imitation_reward = z if imitation_reward is None else imitation_reward
        behaviour_cloning_reward = z if behaviour_cloning_reward is None else behaviour_cloning_reward
        return (self.lambda_task*task_reward + self.lambda_imitation*imitation_reward + self.lambda_contact*contact_reward + self.lambda_bc*behaviour_cloning_reward)
