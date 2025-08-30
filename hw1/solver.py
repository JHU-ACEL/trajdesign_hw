import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Callable, Tuple

class Solver():
    def __init__(self):
        self.tol = 1e-2     # Tolerance check for norm of gradient
        self.max_iter = 100 # Maximum iterations in optimization

    def compute_step_size_ls(self, x0: jnp.array, grad_at_x0: jnp.array, f_eval: Callable) -> float:
        """
        Inputs:
          x0: Initial point from which descent direction is being determined
          grad_at_x0: Gradient at the initial point
          f_eval: function handle for evaluating objective

        Outputs:
          alpha: Step size to take in the direction of the gradient
        """
        raise NotImplementedError("Solver.compute_step_size_ls has yet to be implemented")

    def solve_with_gradient_descent(self, x0: jnp.array, f_eval: Callable) -> Tuple[jnp.array, jnp.array]:
        """
        Inputs:
          x0: Initial point for solver
          f_eval: function handle for evaluating objective

        Outputs:
          x_soln: Optimal solution for this problem
          x_traj: The "trajectory" of the optimizer passed out as an array of size
                (iteration count, x0 dimension) 
        """
        raise NotImplementedError("Solver.solve_with_gradient_descent has yet to be implemented")

    def solve_with_newton_method(self, x0: jnp.array, f_eval: Callable) -> Tuple[jnp.array, jnp.array]:
        """
        Inputs:
          x0: Initial point for solver
          f_eval: function handle for evaluating objective

        Outputs:
          x_soln: Optimal solution for this problem
          x_traj: The "trajectory" of the optimizer passed out as an array of size
                (iteration count, x0 dimension) 
        """
        raise NotImplementedError("Solver.solve_with_newton_method has yet to be implemented")
