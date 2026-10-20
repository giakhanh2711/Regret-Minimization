import importlib
import re
import logging
import numpy as np
import pandas as pd

ALGO = {
    'rm': "Regret Matching",
    'rm+': "Regret Maching+"
}
def load_game_info(game_name: str):
    return importlib.import_module(f"games.{game_name}")

def normalize_cf_cookie(raw):
    raw = raw.strip().strip('"').strip("'")
    match = re.search(r"CF_Authorization=([^;\n]+)", raw)
    return f"CF_Authorization={match.group(1)}" if match else (
        f"CF_Authorization={raw}" if raw.startswith("eyJ") else raw
    )

def create_info_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        f_handler = logging.FileHandler(log_file, mode='w')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(f_handler)

    return logger


def read_payoff_matrix(payoff: dict, actions1: list, actions2: list):
    payoff_matrix = np.zeros((len(actions1), len(actions2), 2))
    for i, a1 in enumerate(actions1):
        for j, a2 in enumerate(actions2):
            payoff_matrix[i, j] = payoff[(a1, a2)]

    return payoff_matrix


def read_empirical_payoff_matrix(filename, strategies: list):
    """
    filename = "data/{game_name}/{game_name}.csv"
    """

    df = pd.read_csv(filename)

    player1 = df.pivot(index='p1_strategy', columns='p2_strategy', values='u1_mean').reindex(index=strategies, columns=strategies)
    player2 = df.pivot(index='p1_strategy', columns='p2_strategy', values='u2_mean').reindex(index=strategies, columns=strategies)

    payoff_matrix = np.stack([player1.values, player2.values], axis=-1)

    return payoff_matrix
