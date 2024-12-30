import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 权重初始化函数
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.apply(initialize_weights)  # 应用权重初始化

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))  # 使用 LeakyReLU
        x = F.leaky_relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.apply(initialize_weights)  # 应用权重初始化

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))  # 使用 LeakyReLU
        x = F.leaky_relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


# PPO Agent
class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=1e-5, eps_clip=0.2, K_epochs=3, update_timestep=15):
        self.env = env
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.update_timestep = update_timestep  # 触发一次更新所需的时间步数
        self.timestep = 0  # 当前时间步数

        self.state_size = env.observation_space.n
        self.action_size = env.action_space.n

        self.policy = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.policy_old = PolicyNetwork(self.state_size, self.action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value_network = ValueNetwork(self.state_size).to(device)
        self.optimizer = optim.Adam([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_network.parameters(), 'lr': lr}
        ])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.99)

        self.MseLoss = nn.MSELoss()

        # 存储轨迹
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

    def select_action(self, state):
        """
        在训练期间使用，选择动作并记录对数概率。
        """
        state = torch.FloatTensor(np.array(state)).to(device)
        action_probs = self.policy_old(state)

        # 检查 action_probs 中是否包含 NaN
        if torch.isnan(action_probs).any():
            raise ValueError("Action probabilities contain NaN values.")

        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def choose_action(self, state):
        """
        在评估和展示期间使用，选择当前策略下的最佳动作。
        """
        state_one_hot = np.zeros(self.state_size)
        state_one_hot[state] = 1
        state_tensor = torch.FloatTensor(state_one_hot).to(device)
        with torch.no_grad():
            action_probs = self.policy(state_tensor)

        # 检查 action_probs 中是否包含 NaN
        if torch.isnan(action_probs).any():
            raise ValueError("Action probabilities contain NaN values during evaluation.")

        action = torch.argmax(action_probs).item()
        return action

    def compute_returns_and_advantages(self, next_value, dones, gamma=0.99, lam=0.95):
        """
        计算广义优势估计 (GAE) 和返回值。
        """
        rewards = self.memory['rewards']
        values = self.memory['values']

        returns = []
        advantages = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                delta = rewards[step] - values[step]
                gae = delta
            else:
                delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
                gae = delta + gamma * lam * (1 - dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]

        returns = torch.FloatTensor(returns).to(device)
        advantages = torch.FloatTensor(advantages).to(device)

        # 使用 unbiased=False 计算标准差
        advantages_mean = advantages.mean()
        advantages_std = advantages.std(unbiased=False)

        # 防止标准差为零
        if advantages_std > 1e-8:
            advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
        else:
            advantages = advantages - advantages_mean

        # 确保返回值和优势的形状一致
        assert returns.shape == advantages.shape, f"Returns shape {returns.shape} and advantages shape {advantages.shape} do not match."

        return returns, advantages

    def update(self):
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(device)
        actions = torch.LongTensor(np.array(self.memory['actions'])).to(device)
        old_log_probs = torch.FloatTensor(np.array(self.memory['log_probs'])).to(device)
        returns = torch.FloatTensor(np.array(self.memory['returns'])).to(device)
        advantages = torch.FloatTensor(np.array(self.memory['advantages'])).to(device)

        for _ in range(self.K_epochs):
            # Get action probabilities
            action_probs = self.policy(states)

            # 检查 action_probs 中是否包含 NaN
            if torch.isnan(action_probs).any():
                raise ValueError("Action probabilities contain NaN values during update.")

            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Get state values
            state_values = self.value_network(states).view(-1)  # 使用 view(-1) 代替 squeeze()

            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss
            value_loss = self.MseLoss(state_values, returns)

            # 确保 state_values 和 returns 的形状一致
            assert state_values.shape == returns.shape, f"State values shape {state_values.shape} and returns shape {returns.shape} do not match."

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy.mean()

            # 检查 loss 是否为 NaN
            if torch.isnan(loss):
                raise ValueError("Loss is NaN.")

            # Take gradient step with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()

            # 检查梯度是否为 NaN
            for param in self.policy.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError("Gradient contains NaN in policy network.")
            for param in self.value_network.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    raise ValueError("Gradient contains NaN in value network.")

            nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=0.5)
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': []
        }

        # 更新学习率
        self.scheduler.step()

    def train(self, episodes=10000, max_timesteps=100):
        print("开始训练...")
        all_rewards = []
        avg_rewards = []
        for episode in range(1, episodes + 1):
            reset_output = self.env.reset()
            if isinstance(reset_output, tuple):
                state, info = reset_output
            else:
                state = reset_output
                info = {}
            done = False
            total_reward = 0

            for t in range(max_timesteps):
                # One-hot encode the state
                state_one_hot = np.zeros(self.state_size)
                state_one_hot[state] = 1
                action, log_prob = self.select_action(state_one_hot)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # 记录状态价值
                state_tensor = torch.FloatTensor(state_one_hot).to(device)
                with torch.no_grad():
                    value = self.value_network(state_tensor).item()

                # 记录到记忆中
                self.memory['states'].append(state_one_hot)
                self.memory['actions'].append(action)
                self.memory['log_probs'].append(log_prob)
                self.memory['rewards'].append(reward)
                self.memory['dones'].append(done)
                self.memory['values'].append(value)

                state = next_state
                total_reward += reward
                self.timestep += 1

                # 当达到更新时间步数时，进行策略更新
                if self.timestep >= self.update_timestep:
                    self.timestep = 0

                    # 计算最后一步的价值估计
                    if done:
                        next_value = 0
                    else:
                        next_state_one_hot = np.zeros(self.state_size)
                        next_state_one_hot[next_state] = 1
                        next_state_tensor = torch.FloatTensor(next_state_one_hot).to(device)
                        with torch.no_grad():
                            next_value = self.value_network(next_state_tensor).item()

                    returns, advantages = self.compute_returns_and_advantages(next_value, self.memory['dones'])
                    self.memory['returns'] = returns.cpu().numpy()
                    self.memory['advantages'] = advantages.cpu().numpy()

                    # 更新策略
                    self.update()

                if done:
                    break

            all_rewards.append(total_reward)

            # 检查是否需要在回合结束时更新策略
            if done and self.timestep > 0:
                # 计算最后一步的价值估计
                if done:
                    next_value = 0
                else:
                    next_state_one_hot = np.zeros(self.state_size)
                    next_state_one_hot[next_state] = 1
                    next_state_tensor = torch.FloatTensor(next_state_one_hot).to(device)
                    with torch.no_grad():
                        next_value = self.value_network(next_state_tensor).item()

                returns, advantages = self.compute_returns_and_advantages(next_value, self.memory['dones'])
                self.memory['returns'] = returns.cpu().numpy()
                self.memory['advantages'] = advantages.cpu().numpy()

                # 更新策略
                self.update()
                self.timestep = 0

            # 每100个回合打印一次平均奖励
            if episode % 100 == 0:
                avg_reward = np.mean(all_rewards[-100:])
                avg_rewards.append(avg_reward)
                print(f"回合 {episode}\t平均奖励: {avg_reward:.2f}")

                # 可选：提前停止训练
                # if avg_reward >= 0.78:
                #     print("环境已解决！")
                #     break

        print("训练完成！")
        return all_rewards, avg_rewards

    def evaluate(self, episodes=100, max_timesteps=100):
        print("开始评估...")
        self.policy.eval()  # 设置为评估模式
        total_rewards = []

        for episode in range(1, episodes + 1):
            reset_output = self.env.reset()
            if isinstance(reset_output, tuple):
                state, info = reset_output
            else:
                state = reset_output
                info = {}
            done = False
            total_reward = 0

            for t in range(max_timesteps):
                action = self.choose_action(state)  # 使用评估时的动作选择方法

                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                total_reward += reward  # 对于 FrozenLake，reward 通常是 0 或 1

                state = next_state

                if done:
                    break

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f"评估完成\t平均奖励: {avg_reward:.2f}")
        self.policy.train()  # 恢复为训练模式
        return avg_reward
