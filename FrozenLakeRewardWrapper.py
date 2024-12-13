import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

class FrozenLakeRewardWrapper(Wrapper):
    def __init__(self, env, step_penalty=-0.03):
        super().__init__(env)
        self.step_penalty = step_penalty

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            desc = self.env.unwrapped.desc.ravel().tolist()
            if desc[state] == b'G':
                reward = 5.0  # 达到目标
            elif desc[state] == b'H':
                reward = -1.0  # 掉入洞穴
        else:
            reward = self.step_penalty  # 每一步的负奖励
        return state, reward, terminated, truncated, info
