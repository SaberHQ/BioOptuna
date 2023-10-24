import optuna

def define_search_space(trial):
    # Define the hyperparameters for Goldrush
    param1 = trial.suggest_float('param1', 0, 1)
    # ... more params for Goldrush ...
    return param1, ...

def objective(params):
    # Evaluate Goldrush with the provided hyperparameters
    # Return the evaluation score
    score = 0
    return score