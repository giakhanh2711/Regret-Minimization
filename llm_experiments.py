from pathlib import Path
import utils
import config
import pandas as pd


from benchmark_experiments import Experiment
from llm_experiment_class import LLMExperiment
from analyst import Analyst
import plot_results
from player import Player

def main():
    game_name = "rps"
    game_module = utils.load_game_info(game_name)

    print("=" * 20, f"Running {game_module.GAME_NAME}", "=" * 20)

    experiment = LLMExperiment(
        game_module.PAYOFFS,
        game_module.STRATEGY_PROMPTS,
        game_name=game_name,
        game_strategies=game_module.GAME_STRATEGIES,
        game_actions=game_module.ACTIONS,
        api_url=config.TAMU_BASE_URL,
        model_name=config.SONNET_MODEL,
        n_reps=config.N_REPS,
    )

    experiment.run_experiment()

    estimate_payoff_file = Path(f"data/{game_name}/{game_name}.csv")
    estimate_payoff_file.parent.mkdir(parents=True, exist_ok=True)
    stats = Analyst(file_name=estimate_payoff_file)
    records = stats.get_records(f"data/{game_name}/{game_name}.json")
    stats.show_estimate_payoff(records)

    # Task 3.4
    emp_payoff_matrix = utils.read_empirical_payoff_matrix(filename=f"data/{game_name}/{game_name}.csv",
                                                           strategies=game_module.GAME_STRATEGIES)
    for algo in utils.ALGO:
        rm_experiment = Experiment(algo=algo,
                                   payoffs=emp_payoff_matrix,
                                   game_name=game_name,
                                   T=5000)
        rm_experiment.run_experiment()


    # Task 3.5
    experiment.run_estimation_noise_experiment(
        n_resamples=10,
        random_seed=2026,
        T=5000,
    )

    df_plus = pd.read_csv(f"data/{game_name}_rmplus.csv")
    NE = game_module.NE
    players = [Player(None, df_plus, i + 1, NE[i]) for i in range(2)]
    
    plot_results.plot_error_bands_strategy(players, game_name)
    plot_results.regret_curve_noise(players, game_name)


if __name__ == "__main__":
    main()