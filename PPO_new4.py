import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

"""
这份代码使用PPO算法完成了4x4地图上FrozenLake游戏，且最终训练出了预期效果
"""


# 将环境包装为向量化环境（适合 PPO 训练）
vec_env = make_vec_env(lambda: gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True), n_envs=1)
#print(vec_env)
#exit()
# 创建 PPO 模型
model = PPO("MlpPolicy", vec_env, device="cpu", verbose=1)

# 训练智能体
print("开始训练智能体...")
model.learn(total_timesteps=26000)
print("训练完成！")


# 创建测试环境（启用渲染）
test_env = make_vec_env(lambda: gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="human"), n_envs=1)
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