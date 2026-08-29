import numpy as np
import torch
from src.ppo.rewards import RewardModule

class BaseEnv:
    """Genesis environment: BaseEnv is supervision; Genesis is the plant."""
    def __init__(self, scene, robot, obj, demo_qpos, contact_tensor, validity_mask,
                 target_pos, target_quat=None, control_dofs=None,
                 reward_module=None, max_steps=None, device=None):
        self.scene, self.robot, self.object = scene, robot, obj
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.demo_qpos = self._tensor(demo_qpos)
        self.contact_tensor = self._tensor(contact_tensor)
        self.validity_mask = torch.as_tensor(validity_mask, dtype=torch.bool, device=self.device)
        self.target_pos = self._tensor(target_pos).reshape(-1)[:3]
        self.target_quat = None if target_quat is None else self._tensor(target_quat).reshape(-1)[:4]
        self.control_dofs = control_dofs if control_dofs is not None else list(range(self.demo_action.shape[-1]))
        self.reward_module = reward_module or RewardModule()
        self.max_steps = max_steps or self.demo_length
        self.initial_qpos = self.demo_qpos[0].clone()
        self.obs_dim = None
        self.frame = 0
        
    def _tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _flat(self, x):
        return self._tensor(x).reshape(-1)

    def reset(self):
        self.frame = 0
        self.robot.set_qpos(self.initial_qpos, zero_velocity=True)
        try:
            self.robot.set_dofs_velocity(torch.zeros(self.robot.n_dofs, device=self.device))
        except Exception:
            pass
        self.scene.step()
        return self.get_observation()

    def get_observation(self):
        obs = torch.cat([
            self._flat(self.robot.get_qpos()),
            self._flat(self.robot.get_vel()),
            self._flat(self.object.get_pos()),
            self._flat(self.object.get_quat()),
            self.target_pos,
        ])
        if self.obs_dim == None:
            self.obs_dim = obs.numel()
        return obs
    
    def get_current_contacts(self):
        K = int(self.contact_tensor.shape[-2])
        out = np.zeros((K, 3), dtype=np.float32)
        try:
            contacts = self.robot.get_contacts(with_entity=self.object)
        except Exception:
            return self._tensor(out)
        if not contacts:
            return self._tensor(out)
        def arr(x):
            if hasattr(x, "detach"): x=x.detach()
            if hasattr(x, "cpu"): x=x.cpu()
            if hasattr(x, "numpy"): return x.numpy()
            return np.asarray(x)
        pos, a, b = arr(contacts.get("position", [])), arr(contacts.get("link_a", [])), arr(contacts.get("link_b", []))
        if pos.size == 0: return self._tensor(out)
        pos, a, b = pos.reshape(-1,3), a.reshape(-1), b.reshape(-1)
        start, end = int(self.robot.link_start), int(self.robot.link_start + self.robot.n_links)
        count = np.zeros(K, dtype=np.int32)
        for i,p in enumerate(pos):
            link = int(a[i]) if i < len(a) and start <= a[i] < end else (int(b[i]) if i < len(b) and start <= b[i] < end else -1)
            local = link - start
            if 0 <= local < K:
                out[local] += p; count[local] += 1
        valid = count > 0
        out[valid] /= count[valid,None]
        return self._tensor(out)

    def get_current_contact_validity(self):
        return torch.linalg.norm(self.get_current_contacts(), dim=-1) > 0

    def compute_reward(self, action):
        t = min(self.frame, self.demo_length-1)
        cr = self.reward_module.compute_contact_reward(self.get_current_contacts(), self.contact_tensor[t,0], self.get_current_contact_validity(), self.validity_mask[t,0])
        br = self.reward_module.compute_behaviour_cloning_reward(action, self.demo_action[t])
        # Contact + BC are enabled by default. Task reward can be added once target orientation/angular state is wired.
        task = torch.ones((), device=self.device) * 0.0
        total = self.reward_module.compute_total_reward(task, cr, None, br)
        return total, {"contact_reward": float(cr.item()), "bc_reward": float(br.item()), "total_reward": float(total.item())}

    

