import gymnasium as gym

from FrozenLakeRewardWrapper import FrozenLakeRewardWrapper
from PPOAgent import PPOAgent
from QLearningAgent import QLearningAgent

# 创建 FrozenLake 环境 (Q-Learning)
training_env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
wrapped_training_env = FrozenLakeRewardWrapper(training_env)

eval_env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True)
wrapped_eval_env = FrozenLakeRewardWrapper(eval_env)

final_env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=True, render_mode = 'human')
wrapped_final_env = FrozenLakeRewardWrapper(final_env)

#agent = QLearningAgent(wrapped_training_env)
agent = PPOAgent(wrapped_training_env)

print("Start training...")
agent.train(episodes=10000)
training_env.close()

agent.env = wrapped_eval_env
print("Start evaluation...")
agent.evaluate(episodes=100)
eval_env.close()

agent.env = wrapped_final_env

# 重置环境，获取初始状态
state, info = wrapped_final_env.reset()

done = False  # 游戏是否结束
step_count = 0  # 计步

while not done:
    wrapped_final_env.render()  # 渲染环境
    action = agent.choose_action(state)  # 选择动作
    state, reward, terminated, truncated, _ = wrapped_final_env.step(action)  # 执行动作
    done = terminated or truncated
    print(f"Step {step_count}: Action={action}, State={state}, Reward={reward}, Done={done}")
    step_count += 1

# 游戏结束后关闭环境
final_env.close()
