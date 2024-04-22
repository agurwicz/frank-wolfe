import os
import inspect
from shutil import rmtree
from datetime import datetime
from json import dump
from itertools import product
from matplotlib import pyplot


class GeneralExperiment:

    def __init__(self, problem_class, experiments, variants, run_experiment_comparison, run_single_variants, output_directory_path):

        self._problem_class = problem_class

        self._output_directory_path = self.__get_output_path(
            output_directory_path=output_directory_path
        )

        self.__run_experiments(
            experiments=self.__get_experiments(
                experiments=experiments
            ),
            variant_experiments=self.__get_variant_experiments(
                variants=variants,
                run_experiment_comparison=run_experiment_comparison,
                run_single_variants=run_single_variants
            )
        )

    def __get_output_path(self, output_directory_path):

        path = os.path.join(
            output_directory_path,
            '{problem}-{time}'.format(
                problem=self._problem_class.__name__.lower(),
                time=datetime.now().strftime(format='%y%m%d_%H%M')
            )
        )

        if os.path.exists(path):
            rmtree(path=path)
        os.makedirs(name=path)

        return path

    def __get_experiments(self, experiments):

        if any(
            parameter not in experiments.keys()
            for parameter, value in inspect.signature(self._problem_class).parameters.items()
            if value.default is inspect.Parameter.empty
        ):
            raise Exception('missing parameter in experiments')

        keys, values = zip(*experiments.items())
        return [dict(zip(keys, value)) for value in product(*values)]

    @staticmethod
    def __get_variant_experiments(variants, run_experiment_comparison, run_single_variants):

        variant_experiments = [variants] if run_experiment_comparison else []

        if run_single_variants:

            for variant, parameters in variants.items():

                if not parameters:
                    variant_experiments.append({variant: parameters})

                else:
                    for parameter in parameters:
                        variant_experiments.append({variant: [parameter]})

        return variant_experiments

    def __run_experiments(self, experiments, variant_experiments):

        def __get_experiment_path(counter):

            path = os.path.join(self._output_directory_path, str(counter))
            os.makedirs(name=path)

            return path

        def __write_experiment_info(path, experiment_data):

            with open(file=os.path.join(path, 'experiment.txt'), mode='w') as file:
                dump(obj=experiment_data, fp=file, indent=4)

        experiment_counter = 0
        for variants in variant_experiments:

            single_variant = (
                True
                if len(variants) == 1 and len(list(variants.values())[0]) <= 1
                else False
            )

            if single_variant:

                experiment_path = __get_experiment_path(counter=experiment_counter)
                experiment_counter += 1

                figure, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 5))

                variant_name = None
                for experiment in experiments:

                    problem = self._problem_class(**experiment)

                    variant_name, parameter = list(variants.items())[0]

                    try:
                        implementation = getattr(problem, variant_name)

                    except AttributeError:
                        raise AttributeError('Variant not implemented')

                    if parameter:
                        parameter = parameter[0]
                        variant_name += '-{parameter}'.format(parameter=parameter)
                        time, errors, times = implementation(parameter)
                    else:
                        time, errors, times = implementation()

                    name = 'size {size}, alpha {alpha}'.format(
                        size=experiment['number_of_features'],
                        alpha=experiment['alpha']
                    )

                    axes[0].semilogy(
                        range(len(errors)),
                        errors,
                        label='{name}: {iterations}'.format(
                            name=name,
                            iterations=len(errors)
                        )
                    )

                    axes[1].semilogy(
                        times,
                        errors,
                        label='{name}: {time:.2E}'.format(
                            name=name,
                            time=time
                        )
                    )

                axes[0].set_xlabel('iteration')
                axes[1].set_xlabel('time')

                for axis in axes:
                    axis.set_ylabel('error/gap')
                    axis.legend()

                pyplot.tight_layout()
                pyplot.subplots_adjust(top=0.92)
                pyplot.suptitle(variant_name)

                pyplot.savefig(os.path.join(experiment_path, 'plot.pdf'))
                pyplot.clf()

            else:

                for index, experiment in enumerate(experiments):

                    experiment_path = __get_experiment_path(counter=experiment_counter)
                    experiment_counter += 1

                    __write_experiment_info(
                        path=experiment_path,
                        experiment_data=experiment,
                    )

                    problem = self._problem_class(**experiment)

                    figure, axes = pyplot.subplots(nrows=1, ncols=2, figsize=(12, 5))

                    try:

                        implementations = {
                            variant_name: (getattr(problem, variant_name), parameters)
                            for variant_name, parameters in variants.items()
                        }

                    except AttributeError:
                        raise AttributeError('One of the variants is not implemented')

                    results = dict()
                    for variant_name, (implementation, parameters) in implementations.items():

                        if not parameters:
                            results[variant_name] = implementation()

                        else:
                            for parameter in parameters:
                                results['{variant}-{parameter}'.format(
                                    variant=variant_name, parameter=parameter
                                )] = implementation(parameter)

                    for name, (time, errors, times) in results.items():

                        axes[0].semilogy(
                            range(len(errors)),
                            errors,
                            label='{name}: {iterations}'.format(
                                name=name,
                                iterations=len(errors)
                            )
                        )

                        axes[1].semilogy(
                            times,
                            errors,
                            label='{name}: {time:.2E}'.format(
                                name=name,
                                time=time
                            )
                        )

                    axes[0].set_xlabel('iteration')
                    axes[1].set_xlabel('time')

                    for axis in axes:
                        axis.set_ylabel('error/gap')
                        axis.legend()

                    pyplot.tight_layout()

                    pyplot.savefig(os.path.join(experiment_path, 'plot.pdf'))
                    pyplot.clf()
