import pandas as pd
import numpy as np

class Player:
    def __init__(self, df, df_plus, i, indv_ne: tuple):
        if df is not None:
            self.df = self.reformat_df(df, i)
        if df_plus is not None:
            self.df_plus = self.reformat_df(df_plus, i)
        self.indv_ne = indv_ne

    def read_nparray(self, df) -> np.ndarray:
        series = df['time-averaged strategy']
        series = series.apply(lambda x: np.array(x.strip('[]').split(), dtype=np.float64))
        return np.stack(series.to_list())
        
    def reformat_df(self, df, i):
        if i == 1 or i == 2:
            if i == 1:
                df = df.rename(columns={"time-averaged strategy 1": "time-averaged strategy",
                                    "average regret 1": "average regret"})
            else:
                df = df.rename(columns={"time-averaged strategy 2": "time-averaged strategy",
                                                    "average regret 2": "average regret"})
            nparray = self.read_nparray(df)
            for j in range(nparray.shape[1]):
                df.insert(len(df.columns), f"action {j + 1}", nparray[:, j])

            self.n_actions = nparray.shape[1]
            return df
 
        print('Only 2 players')
        return df
        