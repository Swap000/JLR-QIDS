import gym
from gym import spaces
import numpy as np

class VehicularEnv(gym.Env):
    def __init__(self):
        super(VehicularEnv, self).__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)

    def reset(self):
        return np.random.uniform(-1, 1, 10)

    def step(self, action):
        state = np.random.uniform(-1, 1, 10)
        reward = np.random.rand()
        done = np.random.rand() > 0.95
        return state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass