from gymnasium import Wrapper

class FrozenLakeRewardWrapper(Wrapper):
    vst = set()
    steps = 0
    def __init__(self, env, step_penalty=0.1):
        super().__init__(env)
        self.vst = set()
        self.steps = 0
        self.step_penalty = step_penalty

    def reset(self, **kwargs):
        self.vst.clear()
        return self.env.reset(**kwargs)

    '''
    def step(self, action): # non slippery
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            desc = self.env.unwrapped.desc.ravel().tolist()
            if desc[state] == b'G':
                reward = 1.0  # 达到目标
            elif desc[state] == b'H':
                reward = -1.0  # 掉入洞穴
            else:
                reward = -1.0   # 超时
        else:
            if state in self.vst:
                reward -= self.step_penalty
            self.vst.add(state)

        return state, reward, terminated, truncated, info
    '''
    def step(self, action): # slippery
        state, reward, terminated, truncated, info = self.env.step(action)
        if terminated or truncated:
            desc = self.env.unwrapped.desc.ravel().tolist()
            if desc[state] == b'G':
                reward = 5.0  # 达到目标
            elif desc[state] == b'H':
                reward = -2.0  # 掉入洞穴
            else:
                reward = -2.0   # 超时
        else:
            reward -= self.step_penalty

        return state, reward, terminated, truncated, info

