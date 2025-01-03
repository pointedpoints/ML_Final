import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv, generate_random_map
from gymnasium.envs.registration import register

"""
这份代码使用PPO算法完成了16x16地图上FrozenLake游戏，且已经训练出了预期效果
和4x4地图上FrozenLake游戏不同的是，在环境中添加了一个访问已经走过的位置的惩罚
"""


# 自定义环境类，添加自定义奖励逻辑
class CustomFrozenLakeEnv(FrozenLakeEnv):
    def __init__(self, desc=None, map_name="16x16", is_slippery=False, max_steps=200, render_mode=None):
        super().__init__(desc=desc, map_name=map_name, is_slippery=is_slippery)
        self.visited_positions = set()  # 记录已访问位置
        self.max_steps = max_steps  # 最大允许步数
        self.step_count = 0  # 当前步数计数
        self.render_mode = render_mode  # 渲染模式

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.step_count += 1

        # 检查是否达到最大步数，设置 truncated
        if self.step_count >= self.max_steps:
            truncated = True

        # 目标和当前位置
        goal_pos = (len(self.desc) - 1, len(self.desc[0]) - 1)  # 目标位置
        agent_pos = self.s // len(self.desc[0]), self.s % len(self.desc[0])  # 智能体当前位置
        distance_to_goal = abs(goal_pos[0] - agent_pos[0]) + abs(goal_pos[1] - agent_pos[1])
        max_distance = len(self.desc) + len(self.desc[0]) - 2

        # 奖励逻辑
        if terminated:
            if agent_pos == goal_pos:  # 到达目标
                reward += 0  # 明确的目标奖励
            else:  # 掉入陷阱
                reward -= 1  # 明确的惩罚
        elif truncated:  # 超时
            reward -= 1  # 超时惩罚
        else:
            # 惩罚重复访问的位置
            if agent_pos in self.visited_positions:
                reward -= 0.1
            self.visited_positions.add(agent_pos)

        # print(f"Step: {self.step_count}, Position: {agent_pos}, Reward: {reward}, Terminated: {terminated}, Truncated:{truncated}")
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.visited_positions.clear()
        self.step_count = 0
        return super().reset(**kwargs)

# 注册自定义环境
register(
    id='FrozenLakeCustom-v1',
    entry_point=__name__ + ':CustomFrozenLakeEnv',
    kwargs={
        'desc': [
            "SFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFHFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFFFHFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFFHFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFHFFFFFFFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFHFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFG"
        ],
        'is_slippery': False,
        "max_steps": 200,  # 最大步数限制
    }
)
register(
    id='FrozenLakeCustom-v2',
    entry_point=__name__ + ':CustomFrozenLakeEnv',
    kwargs={
        'desc': [
            "SFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFHFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFFFHFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFHFFFFFFFFFFF",
            "FFFFFFFFFFFFHFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFHFFFFFFFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFHFFFF",
            "FFFFFFFFFFFFFFFF",
            "FFFFFFFFFFFFFFFG"
        ],
        'is_slippery': False,
        'render_mode': "human",
        "max_steps": 100,  # 最大步数限制
    }
)
map = generate_random_map(size=16)
register(
    id='FrozenLakeCustom-random-no-render',
    entry_point=__name__ + ':CustomFrozenLakeEnv',
    kwargs={
        'desc': map,
        'is_slippery': False,
        "max_steps": 100,  # 最大步数限制
    }
)
register(
    id='FrozenLakeCustom-random-render',
    entry_point=__name__ + ':CustomFrozenLakeEnv',
    kwargs={
        'desc': map,
        'is_slippery': False,
        'render_mode': "human",
        "max_steps": 100,  # 最大步数限制
    }
)

# 将环境包装为向量化环境（适合 PPO 训练）
vec_env = make_vec_env(lambda: gym.make('FrozenLakeCustom-random-no-render'), n_envs=1)

model = PPO(
    "MlpPolicy",
    vec_env,
    device="cpu",
    verbose=0,
    ent_coef=0.1,  # 增强探索
    learning_rate=1e-5,  # 调整学习率
    clip_range=0.3  # 调整策略更新范围
)

# 训练智能体
print("开始训练智能体...")
model.learn(total_timesteps=500000)
print("训练完成！")

# 创建测试环境（启用渲染）
test_env = make_vec_env(lambda: gym.make('FrozenLakeCustom-random-render'), n_envs=1)

# 测试智能体
print("开始测试智能体...")
obs = test_env.reset()  # Gym API
for step in range(100):
    action, _states = model.predict(obs, deterministic=True)  # 确定性策略
    obs, reward, done, info = test_env.step(action)
    test_env.render()  # 渲染环境，观察智能体动作
    if done:
        print(f"游戏结束！奖励：{reward}")
        break
print("测试完成！")
