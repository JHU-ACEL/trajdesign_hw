import numpy as np
import cvxpy as cp

from utils import random_obs, findIC

class MICPSolver():
  """
  Solver for modeling the collision-avoidance problem using mixed-integer programming.

  Attributes:
    nx (int): state dimension.
    nu (int): control dimension.
    N (int): OCP horizon.
    dh (float): time duration for OCP.
    n_obs (int): number of obstacles.
    Ak (np.ndarray): discretized double integrator dynamics.
    Bk (np.ndarray): discretized double integrator dynamics.
    Q (np.ndarray): state error penalty matrix.
    R (np.ndarray): control effort penalty matrix.
    umax (float): minimum L2-norm norm constraint for thruster.
    velmin (float): element-wise minimum constraint for velocity.
    velmax (float): element-wise maximum constraint for velocity.
    posmin (float): element-wise minimum constraint for position.
    posmax (float): element-wise maximum constraint for position.
    n_obs (int): number of obstacles.
    bin_prob (cvxpy problem): cvxpy problem object.
    sampled_params (list): list of params for this problem.
    bin_prob_variables (dict): dict of cvxpy param values;
      keys are self.sampled_params, values are cvxpy.Variable.
    bin_prob_parameters (dict): dict of cvxpy param values;
      keys are self.sampled_params, values are cvxpy.Parameter.
  """
  def __init__(self, N:int, dh:float, n_obs:int, Q:np.ndarray, R:np.ndarray):
      self.nx = 4; self.nu = 2
      nq = int(self.nx / 2)

      self.N = N # horizon
      self.dh = dh

      self.construct_dynamics_matrices()

      self.Q = Q
      self.R = R

      mass_ff = 9.583788668
      thrust_max = 2*1.  # max thrust [N] from two thrusters
      self.umax = thrust_max/mass_ff
      self.velmin = -0.2
      self.velmax = 0.2
      self.posmin = np.zeros(nq)

      ft2m = 0.3048
      self.posmax = ft2m*np.array([12.,9.])

      self.n_obs = n_obs
      self.sampled_params = ['x0', 'xg', 'obstacles']
      return

  def construct_dynamics_matrices(self) -> None:
    """
    Construct double integrator dynamics matrices Ak and Bk
    """
    raise NotImplementedError("RRTSolver. has yet to be implemented")

  def construct_obs(self) -> tuple[np.ndarray, list]:
    """
    Constructs a random set of obstacles and a
      collision free initial state x0.
    Hint: use the helper functions from utils.py.

    Returns:
      x0 (np.ndarray): collision-free initial state.
      obstacles (list): list of n_obs obstacles.
    """
    raise NotImplementedError("RRTSolver.construct_obs has yet to be implemented")

  def init_bin_problem(self, M:float=1e3) -> None:
    """
    Constructs the cvxpy problem.

    Arguments:
      M (float): the big M-value used in the binary
        collision avoidance constraints.
    """
    raise NotImplementedError("RRTSolver.init_bin_problem has yet to be implemented")

  def solve_micp(self, params) -> tuple[bool, float, float, tuple]:
    """High-level method to solve parameterized MICP.

    Args:
      params (dict): Dict of param values; keys are self.sampled_params,
        values are numpy arrays of specific param values.

    Returns:
      prob_success (bool): whether solver succeeded or failed.
      cost: (float): cost for this problem.
      solve_time (float): runtime for this problem.
      primal_solns (tuple[np.ndarray, np.ndarray, np.ndarray]): tuple of
        optimal values for x, u, and y.
    """
    raise NotImplementedError("RRTSolver.solve_micp has yet to be implemented")