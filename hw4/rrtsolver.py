import numpy as np
import matplotlib.pyplot as plt
from utils import plot_line_segments, plot_boxes, line_line_intersection, is_free_state

class RRTSolver():
  """
  Represents a geometric planning problem, where the steering solution between two points is a
  straight line (Euclidean metric).

  This implementation of RRTSolver uses Euclidean distance for nearest neighbor queries
  and straight-line paths for steering between states.

  Attributes:
    x_init (np.ndarray): Initial state.
    x_goal (np.ndarray): Goal state.
    obstacles (list): Obstacle set (line segments).
    configspace_lo (np.ndarray): State space lower bound (e.g., [-5, -5]).
    configspace_hi (np.ndarray): State space upper bound (e.g., [5, 5]).
    eps (float): Maximum steering distance.
    max_iters (int): Maximum number of RRT iterations.
    goal_bias (float): Probability of biasing samples toward the goal.
  """
  def __init__(self, obstacles:list, eps:float, max_iters:int = 1000, goal_bias:float = 0.05):
    raise NotImplementedError("RRTSolver.__init__ has yet to be implemented")

  def set_boundary_conditions(self, x_init:np.ndarray, x_goal:np.ndarray) -> None:
    """Sets boundary conditions x_init and x_goal."""
    raise NotImplementedError("RRTSolver.set_boundary_conditions has yet to be implemented")

  def set_configuration_space_bounds(self, config_lo:np.ndarray, config_hi:np.ndarray) -> None:
    """Sets configuration space bounds."""
    raise NotImplementedError("RRTSolver.set_configuration_space_bounds has yet to be implemented")

  def find_nearest(self, V:np.ndarray, x:np.ndarray) -> int:
    """
    Find the nearest state in V to query state x using Euclidean distance.

    Args:
      V (np.ndarray): Data structure holding the sampled vertices.
      x (np.ndarray): Query state.

    Returns:
      int: Index of nearest point in V to x.
    """
    raise NotImplementedError("RRTSolver.find_nearest has yet to be implemented")

  def steer_towards(self, x:np.ndarray, y:np.ndarray) -> np.ndarray:
    """
    Steer from x towards y along a straight line, bounded by maximum distance eps.

    Args:
      x (np.ndarray): Start state.
      y (np.ndarray): Target state.

    Returns:
      x_steer (np.ndarray): State resulting from bounded steering along straight line.
    """
    raise NotImplementedError("RRTSolver.steer_towards has yet to be implemented")

  def is_free_motion(self, x1:np.ndarray, x2:np.ndarray, n_checks:int=20) -> bool:
    """
    Check if straight-line motion from x1 to x2 is collision-free.

    Args:
      x1 (np.ndarray): Start state of motion.
      x2 (np.ndarray): End state of motion.

    Returns:
      bool: True if motion is collision-free, False otherwise.
    """
    raise NotImplementedError("RRTSolver.is_free_motion has yet to be implemented")

  def plot_tree(self, V:np.ndarray, P:np.ndarray, **kwargs) -> None:
    """
    Plot the RRT tree as straight line segments.

    Args:
      V (np.ndarray): Data structure holding RRT nodes of size (num_samples, config_dim).
      P (np.ndarray): Data structure holding the parent indices for each state of size (num_samples).
      **kwargs: Additional plotting arguments.
    """
    plot_line_segments([(V[P[i],:], V[i,:]) for i in range(V.shape[0]) if P[i] >= 0], **kwargs)

  def plot_path(self, path:list, **kwargs) -> None:
    """
    Plot a path as connected straight line segments.

    Args:
      path (path): List of states representing the path.
      **kwargs: Additional plotting arguments.
    """
    path = np.array(path)
    plt.plot(path[:,0], path[:,1], **kwargs)

  def solve(self) -> None:
    """
    Constructs an RRT rooted at self.x_init with the aim of producing a dynamically-feasible
    and obstacle-free trajectory from self.x_init to self.x_goal.

    Returns:
      None: Function plots the results but doesn't return values officially. See the "Intermediate Outputs"
            descriptions in the implementation for internal data structures.
    """
    ## Intermediate Outputs
    # You must update and/or populate:
    #    - V, P, n: the represention of the planning tree
    #    - succcess: whether or not you've found a solution within max_iters RRT iterations
    #    - solution_path: if success is True, then must contain list of states (tree nodes)
    #          [x_init, ..., x_goal] such that the global trajectory made by linking steering
    #          trajectories connecting the states in order is obstacle-free.

    # if success:
    #   self.plot_path(solution_path, color="green", linewidth=2, label="solution path")
    #   print("Search worked\n")
    # else:
    #   print("Search failed\n")
    # plt.scatter(V[:n,0], V[:n,1])
    # plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
    # plt.annotate(r"$x_{init}$", self.x_init[:2] + [.2, 0], fontsize=16)
    # plt.annotate(r"$x_{goal}$", self.x_goal[:2] + [.2, 0], fontsize=16)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)
    # plt.show()
    raise NotImplementedError("RRTSolver.solve has yet to be implemented")