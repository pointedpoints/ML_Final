import numpy as np
from gymnasium import Wrapper


class FrozenLakeRewardWrapper(Wrapper):
    def __init__(self, env,
                 initial_hole_penalty: float = -10,
                 initial_goal_reward: float = 100,
                 initial_step_penalty: float = 5):
        super().__init__(env)
        self.env = env

        # Set the initial reward values, Goal reward is high and hole penalty is low to encourage the agent to reach the goal
        self.initial_hole_penalty = initial_hole_penalty
        self.initial_goal_reward = initial_goal_reward
        self.initial_step_penalty = initial_step_penalty

        self.hole_penalty = initial_hole_penalty
        self.goal_reward = initial_goal_reward
        self.step_penalty = initial_step_penalty

        self.success_count = 0
        self.hole_count = 0

        self.total_episode = 0
        # set the visit counts for each place
        self.visit_counts = {}
        # initialize the step counter
        self.step_count = 0

    def reset(self):
        self.visit_counts = {}
        self.step_count = 0
        return self.env.reset()

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        if state not in self.visit_counts:
            self.visit_counts[state] = 0
        self.visit_counts[state] += 1

        # Custom reward
        if terminated:
            desc = self.env.unwrapped.desc.ravel().tolist()   # flatten map
            self.total_episode += 1 # one episode is done

            if desc[state] == b'G':# reached the goal
                self.success_count += 1
                reward = self.goal_reward
            elif desc[state] == b'H':# fell into a hole
                self.hole_count += 1
                reward = self.hole_penalty
        else:
            reward = self.step_penalty
            reward += -np.log(self.visit_counts[state]) # penalize the agent for visiting the same state multiple times

        if self.total_episode % 50 == 0 and self.total_episode > 0:
            self.adjust_rewards()

        return state, reward, terminated, truncated, info

    def adjust_rewards(self):
        success_rate = self.success_count / self.total_episode
        print(f"Adjusting rewards: Success rate: {success_rate:.2f}")

        # Adjust the rewards based on the success rate
        self.goal_reward = self.initial_goal_reward * max(0.5, 1 - success_rate) # reduce the goal reward if the success rate is high
        self.hole_penalty = self.initial_hole_penalty * max(0.5, success_rate) * 2 # increase the hole penalty if the success rate is high
        self.step_penalty = self.initial_step_penalty * (1 - 5 * max(0.5, success_rate))  # increase the step penalty if the success rate is high
        print(f"New Rewards -> Goal: {self.goal_reward}, Hole: {self.hole_penalty}, Step: {self.step_penalty}")

