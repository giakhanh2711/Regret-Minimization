GAME_NAME = "Rock-Paper-Scissors"

PAYOFFS = {
    ('ROCK', 'ROCK'): (0, 0),
    ('ROCK', 'PAPER'): (-1, 1),
    ('ROCK', 'SCISSORS'): (1, -1),
    ('PAPER', 'ROCK'): (1, -1),
    ('PAPER', 'PAPER'): (0, 0),
    ('PAPER', 'SCISSORS'): (-1, 1),
    ('SCISSORS', 'ROCK'): (-1, 1),
    ('SCISSORS', 'PAPER'): (1, -1),
    ('SCISSORS', 'SCISSORS'): (0, 0)
}

ACTIONS = ["ROCK", "PAPER", "SCISSORS"]

NE = [(1/3, 1/3, 1/3), (1/3, 1/3, 1/3)]

STRATEGY_PROMPTS = {
    "Uniform": (
        "You are playing rock-paper-scissors. "
        "Ignore the opponent's history actions. Choose rock, paper, or scissors uniformly at random. "
        "End your respond with one word: ROCK, OR PAPER OR SCISSORS"),

    "History": (
        "You are playing rock-paper-scissors. "
        "Try to detect patterns in the opponent's history and "
        "choose the action that you expect to perform best against "
        "the opponent's likely next action. "
        "Output exactly one word: ROCK, OR PAPER OR SCISSORS"),

    "Rock": (
        "You are playing rock-paper-scissors. "
        "Ignore the opponent's history actions. Always choose rock, regardless of the opponent's actions "
        "Output exactly one word: ROCK, OR PAPER OR SCISSORS"),
        
    "Recent": (
        "You are playing rock-paper-scissors. "
        "Play the action that would have beaten the opponent's most recent choice. "
        "If no previous opponent action is available, choose randomly. "
        "Output exactly one word: ROCK, OR PAPER OR SCISSORS"),
        
    "Reason": (
        "Play rock-paper-scissors. "
        "Reason step-by-step about what the opponent might do, then choose and action "
        "that best responds. "
        "Output exactly one word: ROCK, OR PAPER OR SCISSORS"),
}

GAME_STRATEGIES = ["Uniform", "History", "Rock", "Recent", "Reason"]