import optuna

def define_search_space(trial):
    x = trial.suggest_float('x', -10, 10)  # We are searching for x in the range [-10, 10]
    return x,

def objective(params):
    x, = params
    return (x - 2) ** 2