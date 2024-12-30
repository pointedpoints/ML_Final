from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


# 自定义回调类以记录评估期间的奖励
class RewardCallback(BaseCallback):
    """
    自定义回调，用于在训练过程中定期评估模型并记录奖励。
    """
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=100, verbose=1):
        super(RewardCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.rewards = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,  # 在评估时使用确定性策略
                render=False
            )
            self.rewards.append(mean_reward)
            if self.verbose > 0:
                print(f"Step {self.num_timesteps}\t平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")
        return True