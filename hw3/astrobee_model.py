import numpy as np
import casadi as cs
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
import os

class AstrobeeImplicitModel():
  """
  Constructs the AcadosModel used for defining the system dynamics.
  """
  def __init__(self, dt:float=0.1):
    """
    Constructor for the Astrobee model.
    The parameters here are sourced from:
    https://github.com/nasa/astrobee/blob/3fbe361bae9e71ad34d79e8177d54e195fd39800/astrobee/config/worlds/iss.config

    Attributes:
      name (str): model name.
      mass (float): robot mass [kg].
      inertia (np.ndarray): 3x3 inertia tensor [kg*m^2].
      omega_max (float): maximum (and defines minimum) angular rotation [rad/s].
      moment_max (float): maximum moment applied along any axis [N*m].
      dt (float): integration time step.
    """
    self.name = 'astrobee_implicit_model'

    self.mass = 9.583788668
    self.inertia = np.diag([0.15, 0.14, 0.16])
    self.omega_max = 3.1415
    self.moment_max = 0.16 * 0.1750

    self.dt = dt

  def quaternion_kinematic_matrix(self, v: cs.MX) -> cs.MX:
    """
      Parameters:
        v (cs.MX)

      Returns:
        q_kinematics_matrix (cs.MX): 4x4 kinematics matrix.
    """
    raise NotImplementedError("AstrobeeImplicitModel.quaternion_kinematic_matrix has yet to be implemented")

  def get_acados_model(self) -> AcadosModel:
    """
      Constructs the AcadosModel used for the 3-DoF attitude
      planning problem for Astrobee.

      Returns:
        model (AcadosModel): model for Astrobee slewing problem.
    """
    raise NotImplementedError("AstrobeeImplicitModel.get_acados_model has yet to be implemented")
