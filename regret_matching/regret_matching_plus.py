from regret_matching.regret_matching import RegretMatching
import numpy as np

class RegretMatchingPlus(RegretMatching):
    def __init__(self, n_actions, indv_payoff_matrix):
        super().__init__(n_actions, indv_payoff_matrix)

    def flooring(self):
        self.cumulative_regret = np.maximum(self.cumulative_regret, np.zeros(self.n_actions))
        return self.cumulative_regret