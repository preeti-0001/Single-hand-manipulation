import numpy as np

class Action:
    """Represents the action space of the environment."""

    def __init__(self, hand_scale, arm_scale):
        self.hand_scale = hand_scale
        self.arm_scale = arm_scale
        self.actions = np.concatenate([np.zeros(self.hand_scale), np.zeros(self.arm_scale)])
        """Initializes the action space with the given hand and arm scales."""
        

    def get(self):
        """Returns the current action vector, which is a concatenation of hand and arm actions."""
        return self.actions
        


    def split(self, action):
        hand_action = action[..., :self.hand_scale]
        arm_action = action[..., self.hand_scale:self.hand_scale + self.arm_scale]
        
        return hand_action, arm_action

    def to_joint_delta(self, action):
        hand_action, arm_action = self.split(action)

        hand_delta = hand_action * self.hand_scale
        arm_delta = arm_action * self.arm_scale

        return hand_delta, arm_delta

    def apply(self, current_qpos, action):
        """ Returns the new joint positions after applying the action to the current joint positions. """
        hand_delta, arm_delta = self.to_joint_delta(action)

        hand_qpos = current_qpos[..., :self.hand_scale] + hand_delta
        arm_qpos = current_qpos[..., self.hand_scale:self.hand_scale + self.arm_scale] + arm_delta

        return np.concatenate(
            [hand_qpos, arm_qpos],
            axis=-1,
        )


class Observation:
    """Represents the observation space of the environment."""

    def __init__(self, hand_dof=16, arm_dof=6):
        """Initializes the observation space with the given hand and arm degrees of freedom."""
        self.hand_dof = hand_dof
        self.arm_dof = arm_dof
        hand_qpos = np.zeros(self.hand_dof)
        arm_qpos = np.zeros(self.arm_dof)
        object_pose = np.zeros(7) # Quaternion (4) + Position (3)


        self.observation = {
            "hand": hand_qpos,
            "arm": arm_qpos,
            "object": object_pose,
        }

    def get(self):
        return self.observation

    def update(self, hand_qpos, arm_qpos, object_pose):
        self.observation["hand"] = hand_qpos
        self.observation["arm"] = arm_qpos
        self.observation["object"] = object_pose