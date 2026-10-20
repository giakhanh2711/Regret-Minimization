import argparse
import utils
import numpy as np
import pandas as pd
from regret_matching import RegretMatching
from regret_matching_plus import RegretMatchingPlus

class Experiment:
    def __init__(self, algo: str, payoffs: dict, game_name, T: int):
        payoff_matrix = payoffs
        payoff1 = payoff_matrix[:, :, 0]
        payoff2 = payoff_matrix[:, :, 1].T
        n_actions1 = payoff1.shape[0]
        n_actions2 = payoff2.shape[1]
        
        if algo == "rm":
            self.player1 = RegretMatching(n_actions1, payoff1)
            self.player2 = RegretMatching(n_actions2, payoff2)
            self.algo = "rm"
        elif algo == "rm+":
            self.player1 = RegretMatchingPlus(n_actions1, payoff1)
            self.player2 = RegretMatchingPlus(n_actions2, payoff2)
            self.algo = "rmplus"
        else:
            pass

        self.T = T
        self.data = []
        self.joint_action_counts = np.zeros((n_actions1, n_actions2), dtype=int)
        self.emp_joint_dist = np.zeros((n_actions1, n_actions2))
        self.game_name = game_name

    def payoffs_to_matrix(self, payoffs):
        res_comb = np.array(list(payoffs.keys()))
        p1_actions = list(dict.fromkeys(res_comb[:, 0]))
        p2_actions = list(dict.fromkeys(res_comb[:, 1]))
        payoff_matrix = np.zeros((len(p1_actions), len(p2_actions), 2))
        for i, a1 in enumerate(p1_actions):
            for j, a2 in enumerate(p2_actions):
                payoff_matrix[i, j] = np.array(payoffs[(a1, a2)])
        return payoff_matrix
    
    def run_experiment(self):
        for t in range(self.T):
            data_t = {}
            a_t1 = self.player1.make_decision()
            a_t2 = self.player2.make_decision()
            self.update_emp_joint_dist(a_t1, a_t2)
            self.player1.each_round(a_t2)
            self.player2.each_round(a_t1)
            data_t["t"] = t
            data_t["average regret 1"] = self.player1.average_regret()
            data_t["average regret 2"] = self.player2.average_regret()
            data_t["time-averaged strategy 1"] = self.player1.time_averaged_strategy()
            data_t["time-averaged strategy 2"] = self.player2.time_averaged_strategy()

            cce_gap1 = self.player1.cal_cce_violation(self.emp_joint_dist)
            cce_gap2 = self.player2.cal_cce_violation(self.emp_joint_dist.T)
            data_t["cce_gap"] = max(cce_gap1, cce_gap2)

            self.data.append(data_t)
    
        pd.DataFrame(self.data).to_csv(f"data/{self.game_name}_{self.algo}.csv", index=False)

        
    def update_emp_joint_dist(self, a_t1, a_t2):
        self.joint_action_counts[a_t1, a_t2] += 1
        self.emp_joint_dist = self.joint_action_counts / self.joint_action_counts.sum()


def main():
    parser = argparse.ArgumentParser(prog="Running Regret Matching")
    parser.add_argument(
        '-g', '--game_name',
        type=str,
        choices = ['rps', 'asymmetric', 'shapley_game'],
        required = True,
        help='file name to the description of the game'
    )
    parser.add_argument(
        '--T',
        type=int,
        default=10000,
        help="Number of iterations"
    )
    parser.add_argument(
        '-a', '--algo',
        type=str,
        choices=['rm', 'rm+'],
    )

    args = parser.parse_args()

    game_module = utils.load_game_info(args.game_name)
    print(f"{'='* 20} Running {utils.ALGO[args.algo]} for game {game_module.GAME_NAME} {'='*20} \nT = {args.T}")

    exp_runner = Experiment(algo=args.algo,
                            payoffs=game_module.PAYOFFS,
                            game_name=args.game_name,
                            T = args.T)

    exp_runner.run_experiment()


if __name__ == "__main__":
    main()