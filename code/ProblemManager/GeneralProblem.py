from enum import Enum
from time import time
from abc import ABC, abstractmethod
from numpy import zeros, power
from numpy.linalg import norm


class GeneralProblem(ABC):

    def __init__(self, tolerance, max_iterations, x_0, x_size):

        self._tolerance = tolerance
        self._max_iterations = max_iterations

        self._x_0 = x_0 if x_0 else zeros(shape=x_size)
        assert self._check_feasible(x=self._x_0), 'x_0 not in the feasible set'

        self._gradient_lipschitz_constant = self._get_gradient_lipschitz_constant()

        self._errors = None
        self._times = None

    @abstractmethod
    def _get_problem_data(self):
        pass

    @abstractmethod
    def _check_feasible(self, x):
        pass

    @abstractmethod
    def _function(self, x):
        pass

    @abstractmethod
    def _gradient(self, x):
        pass

    @abstractmethod
    def _get_gradient_lipschitz_constant(self):
        pass

    @abstractmethod
    def _linear_oracle(self, x):
        pass

    @abstractmethod
    def _projection(self, x):
        pass

    @abstractmethod
    def _line_search(self, s, x):
        pass

    class __StoppingCondition(Enum):

        iterations = 'maximum iterations reached'
        gap = 'gap is smaller than tolerance'
        error = 'error is smaller than tolerance'

    @staticmethod
    def __show_results(iteration=None, value=None, other=None):

        result = ''

        if iteration and value:
            result += '{iteration}\t{value:.2E}'.format(iteration=iteration, value=value)

        elif other:
            result += other

        print(result)

    @staticmethod
    def optimization_algorithm(func):

        def wrapper(instance, *args):

            instance._errors = list()
            instance._times = list()

            start_time = time()
            func(instance, *args)
            elapsed_time = time() - start_time

            return elapsed_time, instance._errors, [item - start_time for item in instance._times]

        return wrapper

    @optimization_algorithm
    def projected_gradient_descent(self):

        def __step_size():
            return 1/self._gradient_lipschitz_constant

        step_size = __step_size()

        iteration = 0
        x = self._x_0.copy()
        stopping_condition = self.__StoppingCondition.iterations

        while iteration <= self._max_iterations:

            error = self._function(x=x)
            self._errors.append(error)
            self.__show_results(iteration=iteration, value=error)

            if error < self._tolerance:
                self._times.append(time())
                stopping_condition = self.__StoppingCondition.error
                break

            iteration += 1
            x = self._projection(
                x=x - step_size * self._gradient(x=x)
            )

            self._times.append(time())

        self.__show_results(other=stopping_condition.value)

    @optimization_algorithm
    def frank_wolfe_open_loop(self, parameter):

        def __step_size(iteration_index):

            return parameter/(iteration_index + parameter)

        iteration = 0
        x = self._x_0.copy()
        stopping_condition = self.__StoppingCondition.iterations

        while iteration <= self._max_iterations:

            gradient = self._gradient(x=x)

            s = self._linear_oracle(x=gradient)

            gap = gradient.transpose() @ (x-s)
            self._errors.append(gap)
            self.__show_results(iteration=iteration, value=gap)

            if gap < self._tolerance:
                stopping_condition = self.__StoppingCondition.gap
                self._times.append(time())
                break

            step_size = __step_size(iteration_index=iteration)

            iteration += 1
            x = (1-step_size) * x + step_size * s
            self._times.append(time())

        self.__show_results(other=stopping_condition.value)

    @optimization_algorithm
    def frank_wolfe_short_steps(self):

        def __step_size(lipschitz, g_t, d_t):
            return min([
                1,
                g_t/(lipschitz * power(norm(x=d_t, ord=2), 2))
            ])

        iteration = 0
        x = self._x_0.copy()
        stopping_condition = self.__StoppingCondition.iterations

        while iteration <= self._max_iterations:

            gradient = self._gradient(x=x)

            s = self._linear_oracle(x=gradient)
            d = s - x

            gap = -gradient.transpose() @ d
            self._errors.append(gap)
            self.__show_results(iteration=iteration, value=gap)

            if gap < self._tolerance:
                stopping_condition = self.__StoppingCondition.gap
                self._times.append(time())
                break

            step_size = __step_size(
                lipschitz=self._gradient_lipschitz_constant,
                g_t=gap,
                d_t=d
            )

            iteration += 1
            x += step_size * d
            self._times.append(time())

        self.__show_results(other=stopping_condition.value)

    @optimization_algorithm
    def frank_wolfe_line_search(self):

        iteration = 0
        x = self._x_0.copy()
        stopping_condition = self.__StoppingCondition.iterations

        while iteration <= self._max_iterations:

            gradient = self._gradient(x=x)

            s = self._linear_oracle(x=gradient)

            gap = gradient.transpose() @ (x-s)
            self._errors.append(gap)
            self.__show_results(iteration=iteration, value=gap)

            if gap < self._tolerance:
                stopping_condition = self.__StoppingCondition.gap
                self._times.append(time())
                break

            step_size = self._line_search(s=s, x=x)

            iteration += 1
            x = (1-step_size) * x + step_size * s
            self._times.append(time())

        self.__show_results(other=stopping_condition.value)

    @optimization_algorithm
    def frank_wolfe_backtracking(self):

        tau = 2
        eta = 0.9

        def __step_size(cap_l_t, gradient_t, x_t, s_t):

            cap_m = eta * cap_l_t

            while True:

                gamma = min([
                    1,
                    (gradient_t.transpose() @ (x_t - s_t))/(cap_m * power(norm(x=x_t-s_t, ord=2), 2))
                ])

                if self._gradient(x=x_t + gamma*(s-x)).transpose() @ (x-s) >= 0:
                    return gamma, cap_m

                cap_m *= tau

        iteration = 0
        x = self._x_0.copy()

        stopping_condition = self.__StoppingCondition.iterations
        cap_l = None

        while iteration <= self._max_iterations:

            gradient = self._gradient(x=x)

            s = self._linear_oracle(x=gradient)
            d = s - x

            gap = -gradient.transpose() @ d
            self._errors.append(gap)
            self.__show_results(iteration=iteration, value=gap)

            if gap < self._tolerance:
                stopping_condition = self.__StoppingCondition.gap
                self._times.append(time())
                break

            if iteration == 0:
                cap_l = norm(x=gradient-(self._gradient(x=x+self._tolerance*d)), ord=2) / (self._tolerance * norm(x=d, ord=2))

            step_size, cap_l = __step_size(
                cap_l_t=cap_l,
                gradient_t=gradient,
                x_t=x,
                s_t=s
            )

            iteration += 1
            x += step_size * d
            self._times.append(time())

        self.__show_results(other=stopping_condition.value)
