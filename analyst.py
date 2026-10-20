import pandas as pd
import statistics
import json
import math


class Analyst:
    """
    Script that produces the analysis plots and tables.
    file_name = "data/{game_name}/{game_name}.csv"
    """

    def __init__(self, file_name=None):
        self.filename = file_name

    def show_estimate_payoff(self, records: pd.DataFrame):
        # episodes_df = pd.DataFrame(records)
        episodes_df = records
        
        summary_rows = []
        for (s1, s2), group in episodes_df.groupby(["p1_strategy", "p2_strategy"], sort=False):
            u1_mean, u1_ci = self.mean_ci(group["u1"])
            u2_mean, u2_ci = self.mean_ci(group["u2"])
            summary_rows.append({
                "p1_strategy": s1, "p2_strategy": s2, "n": len(group),
                "u1_mean": u1_mean, "u1_ci95": u1_ci,
                "u2_mean": u2_mean, "u2_ci95": u2_ci,
            })
        
        estimates_df = pd.DataFrame(summary_rows)
        print(f"Emperical payoff estimation:\n{estimates_df}")
        if self.filename is not None:
            estimates_df.to_csv(self.filename, index=False)

    # 2.2 Compute the sample mean payoff
    # 2.3 Compute 95% confidence intervals
    def mean_ci(self, values):
        values = list(values)
        mean = statistics.fmean(values)
        half_width = 0.0 if len(values) < 2 else 1.96 * statistics.stdev(values) / math.sqrt(len(values))
        return mean, half_width


    def get_records(self, file_name):
        """
        json file
        """
        with open(file_name, 'r') as f:
            data = json.load(f)
        
        trials = data['trials']
        df_trials = pd.DataFrame(trials)

        return df_trials