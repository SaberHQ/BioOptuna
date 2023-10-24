import optuna

def define_search_space(trial):
    x = trial.suggest_float('x', -10, 10)  # Searching for x in the range [-10, 10]
    y = trial.suggest_float('y', -10, 10)  # Searching for y in the range [-10, 10]
    z = trial.suggest_float('z', -10, 10)  # Searching for z in the range [-10, 10]
    return x, y, z

def objective(params):
    x, y, z = params
    return (x - 3) ** 2 + (y + 2) ** 2 + (z + 1) ** 2

def objective_with_pruning(params):
    x, y, z = params
    intermediate_value = (x - 3) ** 2 + (y + 2) ** 2 + (z + 1) ** 2
    
    # Check if this trial should be pruned (stopped early)
    if intermediate_value > 10:
        raise optuna.TrialPruned()
    return intermediate_value