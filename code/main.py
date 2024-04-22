import sys, os; sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # noqa

from ExperimentManager.GeneralExperiment import GeneralExperiment
from ProblemManager.Lasso import Lasso


if __name__ == '__main__':

    GeneralExperiment(
        problem_class=Lasso,
        experiments={
            'number_of_features': [10, 50, 100, 500],
            'alpha': [0.1],
            'tolerance': [1E-04],
            'max_iterations': [1E05]
        },
        variants={
            'projected_gradient_descent': [],
            'frank_wolfe_open_loop': [1, 2, 3, 4],
            'frank_wolfe_short_steps': [],
            'frank_wolfe_line_search': [],
            'frank_wolfe_backtracking': []
        },
        run_experiment_comparison=True,
        run_single_variants=True,
        output_directory_path=r'experiments'
    )
