import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

# Start with a complex observation space
env = gym.make("CarRacing-v3")
print(env.observation_space.shape)
# (96, 96, 3)  # 96x96 RGB image

# Wrap it to flatten the observation into a 1D array
wrapped_env = FlattenObservation(env)
print(wrapped_env.observation_space.shape)
# (27648,)  # All pixels in a single array
print(wrapped_env.unwrapped.observation_space.shape)
# (96, 96, 3)  # The original observation space

# This makes it easier to use with some algorithms that expect 1D input