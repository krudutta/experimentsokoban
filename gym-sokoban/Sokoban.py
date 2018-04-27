import gym
import gym_sokoban
from gym import spaces
import numpy as np

class SokobanEnv:
    def __init__(self):
        self.env = gym.make('TinyWorld-Sokoban-v1')
        self.action_space      = self.env.action_space
        self.count = 0
        a = self.env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, a[0], a[1]), dtype=np.uint8)

    def step(self, action):
        observation, reward, self.done, _ = self.env.step(action)
        observation = observation.transpose(2, 0, 1)
        return observation, reward, self.done, {}

    def reset(self):
        image = self.env.reset()
        self.done = False
        image = image.transpose(2, 0, 1)
        return image