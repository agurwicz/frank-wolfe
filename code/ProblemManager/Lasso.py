from ProblemManager.GeneralProblem import GeneralProblem

from numpy import zeros, sign, abs, argmax, power
from numpy.linalg import norm
from numpy.random import standard_normal
from jaxopt.projection import projection_l1_ball


class Lasso(GeneralProblem):
    # min ||Ax-b||_2^2 s.t. ||x||_1 <= alpha

    def __init__(self, number_of_features, alpha, tolerance, max_iterations, x_0=None):

        self._number_of_features = number_of_features
        self._alpha = alpha

        self._minimum, self._A, self._b = self._get_problem_data()

        super().__init__(
            tolerance=tolerance,
            max_iterations=max_iterations,
            x_0=x_0,
            x_size=number_of_features
        )

    def _get_problem_data(self):

        def __get_random_feasible_point():

            def __get_point():
                return (self._alpha / self._number_of_features) * standard_normal(size=self._number_of_features)

            point = __get_point()
            while not self._check_feasible(x=point):
                point = __get_point()

            return point

        minimum = __get_random_feasible_point()

        a_matrix = standard_normal(size=(self._number_of_features, self._number_of_features))

        return (
            minimum,
            a_matrix,
            a_matrix @ minimum
        )

    def _check_feasible(self, x):

        return norm(x=x, ord=1) <= self._alpha

    def _function(self, x):

        result = self._A @ x - self._b

        return result.transpose() @ result

    def _gradient(self, x):

        return 2 * self._A.transpose() @ (self._A @ x - self._b)

    def _get_gradient_lipschitz_constant(self):

        return 2 * norm(
            x=self._A.transpose() @ self._A,
            ord=2
        )

    def _linear_oracle(self, x):

        index = argmax(a=abs(x))

        result = zeros(shape=x.shape)
        result[index] = -self._alpha * sign(x[index])

        return result

    def _projection(self, x):

        return projection_l1_ball(
            x=x,
            max_value=self._alpha
        )

    def _line_search(self, s, x):

        q_t = self._A @ (s - x)

        return q_t.transpose() @ (self._b - self._A @ x) / power(norm(x=q_t, ord=2), 2)
