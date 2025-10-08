import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import os

class AstrobeeMPC():
  def __init__(self, model):
    """
    Attributes:
      model (AcadosModel): acados model.
      ocp_solver (AcadosOcpSolver): acados solver.
      N (int): number of time steps in optimal control horizon.
      x0 (np.ndarray): initial state of system.
      Q_mat (list[float]): diagonal elements of the Q cost term.
      R_mat (list[float]): diagonal elements of the R cost term.
    """
    self.model = model
    self.N = 51

    self.x0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # set cost
    self.Q_mat = [1e2, 1e1, 1e1, 1e1]
    self.R_mat = [1e1] * 3

    self.ocp_solver = self.setup(self.x0, self.N)

  def setup(self, x0:np.ndarray, N_horizon:int) -> AcadosOcpSolver:
    """
    Method to set up the AcadosOcpSolver; this is where the constraints and cost terms
    are explicitly defined for the system using the AcadosModel.

    Parameters:
      x0 (np.ndarray): initial state of system. 
      N_horizon (int): number of time steps in optimal control horizon.

    Returns:
      ocp_solver (AcadosOcpSolver): acados solver.
    """
    raise NotImplementedError("AstrobeeMPC.setup has yet to be implemented")

  def solve(self, x0:np.ndarray, verbose:bool=False) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves the optimal control problem starting at an initial state x0.

    Parameters:
      x0 (np.ndarray: initial state.
      verbose (bool): verbosity level of solver.

    Returns:
      x_traj (np.ndarray): state trajectory for system.
      u_traj (np.ndarray): control trajectory for system.
    """
    raise NotImplementedError("AstrobeeMPC.solve has yet to be implemented")
