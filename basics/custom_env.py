import gymnasium as gym
from .gridworld import GridWorldEnv 


gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
    max_episode_steps=300,  # Prevent infinite episodes
)

# Create the environment like any built-in environment
env = gym.make("gymnasium_env/GridWorld-v0")

# Customize environment parameters
env = gym.make("gymnasium_env/GridWorld-v0", size=10)
print(env.unwrapped.size)


# Create multiple environments for parallel training
vec_env = gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)

from gymnasium.utils.env_checker import check_env

# This will catch many common issues
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")

obs, info = env.reset(seed=42)  # Use seed for reproducible testing

env.render()
print(f"Starting position - Agent: {obs['agent']}, Target: {obs['target']}")

# Test each action type
actions = [0, 1, 2, 3]  # right, up, left, down
for action in actions:
    old_pos = obs['agent'].copy()
    obs, reward, terminated, truncated, info = env.step(action)
    new_pos = obs['agent']
    print(f"Action {action}: {old_pos} -> {new_pos}, reward={reward}")

from gymnasium.wrappers import FlattenObservation

# Original observation is a dictionary
env = gym.make('gymnasium_env/GridWorld-v0')
print(env.observation_space)

obs, info = env.reset()
print(obs)

# Wrap it to flatten observations into a single array
wrapped_env = FlattenObservation(env)

print(wrapped_env.observation_space)

print(wrapped_env.unwrapped.observation_space)

obs, info = wrapped_env.reset()
print(obs)

