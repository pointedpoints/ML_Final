import numpy as np
import gymnasium as gym
from gymnasium import Wrapper

class FrozenLakeRewardWrapper(Wrapper):
    vst = set()
    steps = 0
    def __init__(self, env, step_penalty=-0.2):
        super().__init__(env)
        self.vst = set()
        self.steps = 0
        self.step_penalty = step_penalty

    def reset(self, **kwargs):
        self.vst.clear()
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        if terminated:
            desc = self.env.unwrapped.desc.ravel().tolist()
            if desc[state] == b'G':
                reward = 20.0  # 达到目标
            elif desc[state] == b'H':
                reward = -10.0  # 掉入洞穴
            else:
                reward = -50.0   # 超时
        else:
            if state in self.vst:
                reward = self.step_penalty
            else:
                self.vst.add(state)
                reward = 0.5

        return state, reward, terminated, truncated, info
