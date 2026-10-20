import numpy as np


class RegretMatching:
    def __init__(self, n_actions, indv_payoff_matrix):
        self.n_actions = n_actions
        self.cumulative_regret = np.zeros(n_actions)
        self.fixed_action_payoff = np.zeros(n_actions)
        self.mixed_strategy_payoff = 0
        self.cumulative_strategy = np.zeros(n_actions)
        self.sigma = np.ones(n_actions) / n_actions
        self.indv_payoff = indv_payoff_matrix
        self.T = 0

    def each_round(self, opponent_move):
        self.cumulative_strategy += self.sigma
        self.T += 1
        payoff_t = np.dot(self.sigma, self.indv_payoff[:, opponent_move])
        round_regrets = self.indv_payoff[:, opponent_move] - payoff_t
        self.cumulative_regret += round_regrets
        self.update_sigma()

    def update_sigma(self):
        tmp = self.flooring()
        if np.allclose(tmp, 0):
            self.sigma = np.ones(self.n_actions) / self.n_actions
        else:
            self.sigma = tmp / tmp.sum()

    def make_decision(self):
        return np.random.choice(np.arange(self.n_actions), p=self.sigma)

    def flooring(self):
        return np.maximum(self.cumulative_regret, np.zeros(self.n_actions))

    def time_averaged_strategy(self):
        return self.cumulative_strategy / self.T

    def average_regret(self):
        return max(0.0, np.max(self.cumulative_regret)) / self.T

    def cal_cce_violation(self, emp_joint_dist):
        avg_payoff = np.sum(self.indv_payoff * emp_joint_dist)
        opponent_dist = emp_joint_dist.sum(axis=0)
        deviation_payoffs = self.indv_payoff @ opponent_dist
        return max(0.0, np.max(deviation_payoffs - avg_payoff))