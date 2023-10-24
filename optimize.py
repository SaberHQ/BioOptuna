import optuna
from optuna.trial import TrialState
import argparse
import sys

# Importing tool modules
from tools import goldrush, test

def parse_arguments():
    
    parser = argparse.ArgumentParser(description="Optimize hyperparameters for bioinformatics tools using Optuna.")

    subparsers = parser.add_subparsers(dest='mode', required=True, help='Choose a tool to optimize hyperparameters for.')

    # Shared arguments
    def add_shared_arguments(subparser):
        subparser.add_argument("--sampler", type=str, choices=['random', 'tpe', 'cmaes'], default='tpe', help='Sampler to be used for hyperparameter optimization.')
        subparser.add_argument("--pruner", type=str, choices=['none', 'median', 'halving', 'successive', 'hyperband'], default='hyperband', help='Pruner to be used for hyperparameter optimization.')
        subparser.add_argument("-n", "--n_trials", type=int, default=100, help="Number of trials for optimization. Default is 100.")
        subparser.add_argument('--seed', type=int, default=192, help='Random seed for reproducibility.')
        subparser.add_argument("-d", "--direction", type=str, choices=['minimize', 'maximize'], default='minimize', help="Direction of optimization. Default is 'minimize'.")
        subparser.add_argument("--storage", type=str, default=None, help="Database URL for Optuna. (default='None'). If you're running experiments that you don't wish to persist, consider using Optuna's in-memory storage: 'sqlite:///:memory:', otherwise select a db name: 'sqlite:///goldrush_optuna.db' for exsample.")
        subparser.add_argument("-s", "--study_name", type=str, default="biooptuna_study", help="Name of the Optuna study. Default is 'biooptuna_study'.")

    # Goldrush subparser
    goldrush_parser = subparsers.add_parser('goldrush', help='Optimize hyperparameters for Goldrush.')
    add_shared_arguments(goldrush_parser)

    # Test subparser
    test_parser = subparsers.add_parser('test', help='Optimize hyperparameters for test function.')
    add_shared_arguments(test_parser)

    # Add more subparser as needed (e.g. for other bioinformatics tools developed within BTL lab)
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
    
    return parser.parse_args()


def main(args):

    # Prinout the mode, arguments and their values
    print(f"Optimizing hyperparameters for {args.mode} with the following arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Depending on the mode, select the appropriate tool module
    if args.mode == 'goldrush':
        tool_module = goldrush
    elif args.mode == 'test':
        tool_module = test
    
    # ... handle other sub-commands ...
    
    else:
        raise ValueError("Invalid mode. Please specify a valid mode.")
    
    # Define the study and objective
    def objective(trial):
        params = tool_module.define_search_space(trial)
        return tool_module.objective(params)

    study = optuna.create_study(direction=args.direction)  # Adjust direction as needed
    study.optimize(objective, n_trials=args.n_trials)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == "__main__":
    args = parse_arguments()
    main(args)