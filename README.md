# Regret-Minimization
Regret Minimization over LLM-Generated Strategies

### To run Task 2 for 3 game with RM or RM+
python benchmark\_experiments.py -g <game_name> -a <algo_name>

<game_name> = {rps, asymmetric, shapley_game}
<algo_name> = {rm, rm+}

### To plot results from RM and RM+
python plot\_results.py -g <game_name>

### To run Task 3 LLM-Generated Strategies as Actions (rps game default)
python llm_experiment.py

