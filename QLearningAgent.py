from collections import defaultdict

import gymnasium as gym
import numpy as np

class QLearningAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.1,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        final_epsilon: float = 0.01,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict( lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def choose_action(self, obs: int) -> int :
        """Choose an action using an epsilon-greedy policy.

        Args:
            obs: The current state

        Returns:
            The action to take
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()# randomly choose an action
        else:
            return int(np.argmax(self.q_values[obs]))# choose the best action

    def choose_best_action(self, obs: int) -> int:
        """Choose the best action according to the Q-value table.

        Args:
            obs: The current state

        Returns:
            The best action to take
        """
        return int(np.argmax(self.q_values[obs]))

    def update_Q_table(self, state: int,
                       action: int,
                       reward: float,
                       terminated: bool,
                       next_state: int) -> None:
        """Update the Q-value table using the Q-learning update rule.

        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
        """
        future_Q = (not terminated) * np.max(self.q_values[next_state])
        td_target = reward + self.discount_factor * future_Q - self.q_values[state][action]
        self.q_values[state][action] += self.lr * td_target
        self.training_error.append(td_target)

    def decay(self):
        """Decay the epsilon value."""
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, episodes:int = 1000) -> None:
        """Train the agent for a number of episodes.

        Args:
            episodes: The number of episodes to train for
        """
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_obs, reward, terminated, truncated, info= self.env.step(action)
                self.update_Q_table(state, action, float(reward) , terminated, next_obs)
                done = terminated or truncated
                state = next_obs
                total_reward += reward

            self.decay()
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}/{episodes} - Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}")


    def evaluate(self, episodes: int = 100) -> None:
        """Evaluate the agent over a number of episodes.

        Args:
            episodes: The number of episodes to evaluate over

        Returns:
            The average reward per episode
        """
        success = 0
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False

            while not done:
                action = self.choose_best_action(state)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                state = next_obs
                if reward > 0:
                    success += 1

        print(f"Success Rate: {success / episodes:.2f}")