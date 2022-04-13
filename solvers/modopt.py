from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from modopt.opt import gradient, proximity, algorithms, cost


class Solver(BaseSolver):
    """Gradient descent solver using modopt"""

    name = "Modopt"

    def set_objective(self, X, y, fit_intercept=False):
        self.X, self.y = X, y
        self.fit_intercept = fit_intercept

    def run(self, n_iter):
        grad_op = gradient.GradBasic(
            self.y, lambda a: np.dot(self.X, a), lambda a: np.dot(self.X.T, a)
        )
        prox_op = proximity.IdentityProx()

        x_0 = np.zeros(self.X.shape[1])
        eta = 1 / np.linalg.norm(self.X, ord=2) ** 2

        alg = algorithms.VanillaGenericGradOpt(
            x_0,
            grad_op,
            prox_op,
            cost=cost.costObj([grad_op, prox_op], tolerance=1e-30, cost_interval=None),
            eta=eta,
        )

        alg.iterate(max_iter=n_iter)

        self.w = alg.x_final

    def get_result(self):
        return self.w
