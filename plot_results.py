import pandas as pd
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import argparse
from pathlib import Path
import utils
from player import Player

def regret_curve(players, game_name):
    fig, ax = plt.subplots(nrows=1, ncols=len(players), figsize=(14, 5))
    T = np.arange(1, 10**4 + 1)
    for i in range(len(players)):
        ax[i].plot(players[i].df['average regret'], c='g', label="regret matching")
        ax[i].plot(players[i].df_plus['average regret'], c = 'y', label='regret matching+')
        ax[i].plot(T, 1 / np.sqrt(T), c='r', label="O(1/sqrt(T))")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel("T")
        ax[i].set_ylabel("Average regret")
        ax[i].set_title(f"Regret curve for player {i + 1}\n{game_name}")
        ax[i].legend()

    plt.savefig(f"img/{game_name}/regret_curve.png")


def strategy_convergence(players, game_name):
    n_actions = [p.n_actions for p in players]
    fig = plt.figure(layout='constrained', figsize=(17, 9))
    fig.suptitle("Strategy Convergence", fontsize='xx-large')
    subfigs = fig.subfigures(nrows=len(players), ncols=1, hspace=0.1)

    for i in range(len(players)):
        subfigs[i].suptitle(f"Player {i + 1} - {game_name}", fontsize='x-large')
        ax = subfigs[i].subplots(nrows=1, ncols=max(n_actions))
        for j in range(players[i].n_actions):
            ax[j].plot(players[i].df[f"action {j + 1}"], c='g', label="regret matching")
            ax[j].plot(players[i].df_plus[f"action {j + 1}"], c = 'y', label='regret matching+')
            ax[j].axhline(y=players[i].indv_ne[j], c='r', label='equilibrium')
            ax[j].set_xscale('log')
            ax[j].set_xlabel("T")
            ax[j].set_ylabel("Mixed strategy")
            ax[j].set_title(f"Mixed strategy convergence for action {j + 1}")
            ax[j].legend()

    plt.savefig(f"img/{game_name}/strategy_convergence.png")

def plot_cce_gap(players, game_name):
    fig, ax = plt.subplots(nrows=1, ncols=len(players), figsize=(14, 5))
    for i in range(len(players)):
        ax[i].plot(players[i].df['cce_gap'], c='g', label="regret matching")
        ax[i].plot(players[i].df_plus['cce_gap'], c = 'y', label='regret matching+')
        ax[i].set_xscale('log')
        ax[i].set_xlabel("T")
        ax[i].set_ylabel("CCE gap")
        ax[i].set_title(f"Maximum CCE violation {i + 1}\n{game_name}")
        ax[i].legend()
    plt.savefig(f"img/{game_name}/cce_gap.png")

def plot_error_bands_strategy(players, game_name):
    n_actions = [p.n_actions for p in players]
    fig = plt.figure(layout='constrained', figsize=(17, 9))
    fig.suptitle("Strategy Convergence", fontsize='xx-large')
    subfigs = fig.subfigures(nrows=len(players), ncols=1, hspace=0.1)

    
    for i in range(len(players)):
        subfigs[i].suptitle(f"Player {i + 1}\n{game_name}", fontsize='x-large')
        ax = subfigs[i].subplots(nrows=1, ncols=max(n_actions))
        for j in range(players[i].n_actions):
            
            for k in range(10):
                noise_path = noise_path = (
                    Path("data") / game_name / "estimation_noise"
                    / f"resample_{k + 1}_rmplus.csv"
                )
                df_plus = pd.read_csv(noise_path)
                player_sample = Player(df=None, df_plus=df_plus, i=i+1, indv_ne=(1/3, 1/3, 1/3))
                ax[j].plot(player_sample.df_plus[f"action {j + 1}"], label=f'RM+ resampled {k + 1}')

            ax[j].plot(players[i].df_plus[f"action {j + 1}"], c = 'y', label='regret matching+')
            # ax[j].axhline(y=players[i].indv_ne[j], c='r', label='equilibrium')
            ax[j].set_xscale('log')
            ax[j].set_xlabel("T")
            ax[j].set_ylabel("Mixed strategy")
            ax[j].set_title(f"Measure the effect of estimation noise")
            ax[j].legend()

    Path(f"img/{game_name}/estimation_noise").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"img/{game_name}/estimation_noise/strategy_convergence_noise.png")


def regret_curve_noise(players, game_name):
    fig, ax = plt.subplots(nrows=1, ncols=len(players), figsize=(14, 5))
    T = np.arange(1, 10**4 + 1)

    for i in range(len(players)):
        for k in range(10):
            noise_path = (
                Path("data") / game_name / "estimation_noise"
                / f"resample_{k + 1}_rmplus.csv"
            )
            df_plus = pd.read_csv(noise_path)
            player_sample = Player(df=None, df_plus=df_plus, i=i+1, indv_ne=(1/3, 1/3, 1/3))
            ax[i].plot(player_sample.df_plus["average regret"], label=f'RM+ resampled {k + 1}')

        ax[i].plot(players[i].df_plus['average regret'], c = 'y', label='regret matching+')
        ax[i].plot(T, 1 / np.sqrt(T), c='r', label="O(1/sqrt(T))")
        ax[i].set_xscale('log')
        ax[i].set_yscale('log')
        ax[i].set_xlabel("T")
        ax[i].set_ylabel("Average regret")
        ax[i].set_title(f"Regret curve (resampled) for player {i + 1}\n{game_name}")
        ax[i].legend()

    Path(f"img/{game_name}/estimation_noise").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"img/{game_name}/estimation_noise/regret_curve.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-g', '--game_name',
        type=str,
        choices = ['rps', 'asymmetric', 'shapley_game'],
        required=True,
    )

    args = parser.parse_args()
    game_name = args.game_name
    game_module = utils.load_game_info(game_name)
    NE = game_module.NE

    df = pd.read_csv(f"data/{game_name}_rm.csv")
    df_plus = pd.read_csv(f"data/{game_name}_rmplus.csv")

    players = [Player(df, df_plus, i + 1, NE[i]) for i in range(2)]

    img_dir = Path("img") / f"{game_name}"
    img_dir.mkdir(parents=True, exist_ok=True)

    regret_curve(players, game_name)
    strategy_convergence(players, game_name)
    plot_cce_gap(players, game_name)
    plt.show()


if __name__ == "__main__":
    main()



