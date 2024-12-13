import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        action_probs = torch.softmax(self.fc3(x), dim=-1)
        return action_probs


# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        state_value = self.fc3(x)
        return state_value


# PPO Agent
class PPOAgent:
    def __init__(self, env, gamma=0.95, lr=1e-4, eps_clip=0.2, K_epochs=10, update_timestep=500):
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
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_network.parameters()), lr=lr)

        self.MseLoss = nn.MSELoss()

        # 存储轨迹
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

    def select_action(self, state):
        """
        在训练期间使用，选择动作并记录对数概率。
        """
        state = torch.FloatTensor( np.array(state) ).to(device)
        action_probs = self.policy_old(state)
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
        action = torch.argmax(action_probs).item()
        return action

    def compute_returns(self, rewards, dones, next_value, lam=0.95):
        """
                计算广义优势估计 (GAE)
                """
        returns = []
        gae = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                next_value = 0
            delta = rewards[step] + self.gamma * (1 - dones[step]) * next_value - self.value_network(
                torch.FloatTensor(self.memory['states'][step]).to(device)).item()
            gae = delta + self.gamma * lam * (1 - dones[step]) * gae
            returns.insert(0,
                           gae + self.value_network(torch.FloatTensor(self.memory['states'][step]).to(device)).item())
            next_value = self.value_network(torch.FloatTensor(self.memory['states'][step]).to(device)).item()
        return returns

    def update(self):
        # Convert lists to tensors
        states = torch.FloatTensor( np.array(self.memory['states']) ).to(device)
        actions = torch.LongTensor( np.array(self.memory['actions']) ).to(device)
        old_log_probs = torch.FloatTensor( np.array(self.memory['log_probs']) ).to(device)
        returns = torch.FloatTensor( np.array(self.memory['returns']) ).to(device)
        advantages = returns - self.value_network(states).detach().squeeze()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.K_epochs):
            # Get action probabilities
            action_probs = self.policy(states)
            dist = Categorical(action_probs)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()

            # Get state values
            state_values = self.value_network(states).squeeze()

            # Calculate ratios
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Calculate value loss
            value_loss = self.MseLoss(state_values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss - 0.02 * entropy.mean()

            # Take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # 清空记忆
        self.memory = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': []
        }

    def train(self, episodes=10000, max_timesteps=100):
        print("开始训练...")
        all_rewards = []
        avg_rewards = []
        for episode in range(1, episodes + 1):
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for t in range(max_timesteps):
                # One-hot encode the state
                state_one_hot = np.zeros(self.state_size)
                state_one_hot[state] = 1
                action, log_prob = self.select_action(state_one_hot)

                next_state, reward, terminated, truncated, info = self.env.step(action)

                # 记录到记忆中
                self.memory['states'].append(state_one_hot)
                self.memory['actions'].append(action)
                self.memory['log_probs'].append(log_prob)
                self.memory['rewards'].append(reward)
                self.memory['dones'].append(terminated or truncated)

                state = next_state
                total_reward += reward
                self.timestep += 1

                # 当达到更新时间步数时，进行策略更新
                if self.timestep >= self.update_timestep:
                    self.timestep = 0

                    # 计算最后一步的价值估计
                    state_one_hot_tensor = torch.FloatTensor(np.zeros(self.state_size)).to(device)
                    state_one_hot_tensor[state] = 1
                    with torch.no_grad():
                        next_value = self.value_network(state_one_hot_tensor).item()
                    returns = self.compute_returns(self.memory['rewards'], self.memory['dones'], next_value)
                    self.memory['returns'] = returns

                    # 更新策略
                    self.update()

            all_rewards.append(total_reward)

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
            state, info = self.env.reset()
            done = False
            total_reward = 0

            for t in range(max_timesteps):
                action = self.choose_action(state)  # 使用评估时的动作选择方法

                next_state, reward, terminated, truncated, info  = self.env.step(action)
                total_reward += reward
                state = next_state
                if reward>0:
                    total_reward = 1

                if terminated or truncated:
                    break

            total_rewards.append(total_reward)

        avg_reward = np.mean(total_rewards)
        print(f"评估完成\t平均奖励: {avg_reward:.2f}")
        self.policy.train()  # 恢复为训练模式
        return avg_reward
