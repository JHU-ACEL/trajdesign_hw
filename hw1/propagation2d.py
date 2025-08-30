# propagation2d.py
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Propagation2D:
  """
  2D double-integrator propagation with Euler, RK4, exact ZOH (discrete),
  and Euler–Maruyama (stochastic) simulation.

  State: [x, y, vx, vy]
  Input: [ax, ay]
  """
  def __init__(self, A, B) -> None:
    # Continuous-time system (kept for reference/extension).
    self.A = A
    self.B = B

  # ----------- Builders -------------------------------------------------

  @staticmethod
  def state_t(p_x: float, p_y: float, v_x: float, v_y: float) -> jnp.ndarray:
    """Create a state vector [x, y, vx, vy]."""
    return jnp.array([p_x, p_y, v_x, v_y])

  @staticmethod
  def input_t(a_x: float, a_y: float) -> jnp.ndarray:
    """Create an input vector [ax, ay]."""
    return jnp.array([a_x, a_y])

    # ----------- Dynamics -------------------------------------------------

  @staticmethod
  def f(state: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
    """
    Continuous-time 2D double-integrator dynamics.

    Args:
      state: [x, y, vx, vy] -> jnp.array
      u_t:   [ax, ay] -> jnp.array

    Returns:
      [vx, vy, ax, ay] -> jnp.array
    """
    raise NotImplementedError("Propagation2D.f has yet to be implemented")

    # ----------- Euler ----------------------------------------------------

  def euler_step(
      self, state: jnp.ndarray, u_t: jnp.ndarray, dt: float
  ) -> jnp.ndarray:
    """One Euler step for the double integrator."""
    raise NotImplementedError("Propagation2D.euler_step  has yet to be implemented")

  def euler_normal(
    self,
    state0: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    total_time: float,
  ) -> np.ndarray:
    """
    Propagate with Euler integration under constant input u.

    Returns:
      (N+1) x 5 array: [x, y, vx, vy, t]
    """
    raise NotImplementedError("Propagation2D.euler_normal has yet to be implemented")

    # ----------- RK4 ------------------------------------------------------

  def rk4_step(
      self, state: jnp.ndarray, u_t: jnp.ndarray, dt: float
  ) -> jnp.ndarray:
    """One RK4 step for the double integrator."""
    raise NotImplementedError("Propagation2D.rk4_step has yet to be implemented")

  def rk4_normal(
    self,
    state0: jnp.ndarray,
    u: jnp.ndarray,
    dt: float,
    total_time: float,
  ) -> np.ndarray:
    """
    Propagate with RK4 integration under constant input u.

    Returns:
      (N+1) x 5 array: [x, y, vx, vy, t]
    """
    raise NotImplementedError("Propagation2D.rk4_normal has yet to be implemented")

    # ----------- Exact ZOH (discrete) ------------------------------------

  @staticmethod
  def c2d_double_integrator_closed_form(dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact ZOH discretization (closed-form) for the 2D double integrator.

    Returns:
      Phi (4x4), Gamma (4x2)
    """
    raise NotImplementedError("Propagation2D.c2d_double_integrator_closed_form has yet to be implemented")

  def analytical_solution(
    self,
    dt: float,
    total_time: float,
    x0: jnp.ndarray,
    u_seq: Iterable[float] | jnp.ndarray | np.ndarray,
  ) -> np.ndarray:
    """
    Simulate x_{k+1} = Phi x_k + Gamma u_k with exact ZOH.

    Args:
        dt: time step
        total_time: total simulation time
        x0: initial state, shape (4,)
        u_seq: either a single (2,) constant input or a sequence (N, 2)

    Returns:
        (N+1) x 5 trajectory array: [x, y, vx, vy, t]
    """
    raise NotImplementedError("Propagation2D.analytical_solution has yet to be implemented")

    # ----------- Continuous-time exact final (optional helper) -----------

  @staticmethod
  def exact_final_state(
      state0: jnp.ndarray, u: jnp.ndarray, T: float
  ) -> np.ndarray:
    """
    Closed-form continuous-time final state under constant acceleration u.

    Returns:
        length-5 array: [x, y, vx, vy, t]
    """
    raise NotImplementedError("Propagation2D.final_state has yet to be implemented")

    # ----------- Stochastic (Euler–Maruyama) -----------------------------

  @staticmethod
  def em_step(
    state: np.ndarray,
    u: np.ndarray,
    dt: float,
    sigma: Iterable[float] | np.ndarray,
    rng: np.random.Generator,
  ) -> np.ndarray:
    """
    One Euler–Maruyama step for the 2D stochastic double integrator.

    Args:
      state: (4,) [x, y, vx, vy]
      u:     (2,) [ax, ay]
      dt:    time step
      sigma: (2,) acceleration noise std devs [sigma_x, sigma_y]
      rng:   NumPy random Generator

    Returns:
      next state (4,)
    """
    raise NotImplementedError("Propagation2D.em_step has yet to be implemented")

  def simulate_em(
    self,
    x0: np.ndarray,
    u_func: Callable[[np.ndarray, float], np.ndarray],
    T: float,
    dt: float,
    sigma: Iterable[float] | np.ndarray,
    seed: int,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate ONE trajectory with Euler–Maruyama.

    Args:
      x0: initial state (4,)
      u_func: function(state, t) -> control (2,)
      T: total time
      dt: time step
      sigma: (2,) acceleration noise std devs
      seed: RNG seed

    Returns:
      t:    (N+1,)
      traj: (N+1, 4)
    """
    raise NotImplementedError("Propagation2D.simulate_em has yet to be implemented")

  def simulate_em_paths_by_seeds(
    self,
    x0: np.ndarray,
    u_func: Callable[[np.ndarray, float], np.ndarray],
    T: float,
    dt: float,
    sigma: Iterable[float] | np.ndarray,
    seeds: Iterable[int],
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate many EM paths by using a different RNG seed for each path.

    Returns:
      t: (N+1,)
      X: (N+1, n_paths, 4)
    """
    raise NotImplementedError("Propagation2D.simulate_em_paths_by_seed has yet to be implemented")

    # ----------- Plotting -------------------------------------------------

  @staticmethod
  def plot_traj(
    trajectories: Iterable[np.ndarray],
    labels: Iterable[str] | None = None,
    markers: Iterable[str] | None = None,
  ) -> None:
    """
    Plot one or more deterministic trajectories.
    Each trajectory is shaped (N+1, 5): [px, py, vx, vy, t].
    """
    trajectories = list(trajectories)
    if labels is None:
      labels = [f"traj_{i + 1}" for i in range(len(trajectories))]
    else:
      labels = list(labels)

    if markers is None:
      base = ["o", "s", "d", "^", "v", "x", "+"]
      reps = (len(trajectories) // len(base)) + 1
      markers = (base * reps)[: len(trajectories)]
    else:
      markers = list(markers)

    # XY plot
    plt.figure()
    for traj, label, marker in zip(trajectories, labels, markers):
      px, py, vx, vy, t = traj.T
      plt.plot(px, py, marker=marker, markersize=2, linewidth=1, label=label)
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.title("2D Double Integrator Trajectories")
    plt.axis("equal")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Position vs time
    plt.figure()
    for traj, label in zip(trajectories, labels):
      px, py, vx, vy, t = traj.T
      plt.plot(t, px, label=f"px_{label}")
      plt.plot(t, py, label=f"py_{label}")
    plt.xlabel("time [s]")
    plt.ylabel("position")
    plt.title("Position vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Velocity vs time
    plt.figure()
    for traj, label in zip(trajectories, labels):
      px, py, vx, vy, t = traj.T
      plt.plot(t, vx, label=f"vx_{label}")
      plt.plot(t, vy, label=f"vy_{label}")
    plt.xlabel("time [s]")
    plt.ylabel("velocity")
    plt.title("Velocity vs Time")
    plt.legend()
    plt.grid(True)
    plt.show()

  @staticmethod
  def plot_error(error_euler: np.ndarray, error_rk4: np.ndarray) -> None:
    """
    Plot bar graph of errors for Euler vs RK4.
    Only the first two (position) entries are used, matching prior code.
    """
    x_euler = error_euler[0]
    x_rk4 = error_rk4[0]
    y_euler = error_euler[1]
    y_rk4 = error_rk4[1]

    values = [x_euler, x_rk4, y_euler, y_rk4]
    labels = ["x_euler", "x_rk4", "y_euler", "y_rk4"]

    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x, values, width=0.6)
    ax.set_ylabel("Error")
    ax.set_title("Euler vs RK4 Errors")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45)

    for i, v in enumerate(values):
      ax.text(i, v + 0.01 * max(values), f"{v:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.show()

    # ----- Stochastic plotting helpers (for EM) --------------------------

  @staticmethod
  def plot_xy_trajectories(trajs: np.ndarray, title: str) -> None:
    """
    Plot many stochastic trajectories in the xy-plane.

    Args:
      trajs: (N+1, n_paths, 4)
    """
    plt.figure(figsize=(6, 6))
    n_paths = trajs.shape[1]
    for k in range(n_paths):
      plt.plot(trajs[:, k, 0], trajs[:, k, 1], linewidth=1, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()

  @staticmethod
  def plot_mean_with_percentile_band(
    t: np.ndarray,
    values: np.ndarray,
    p_lo: float = 5,
    p_hi: float = 95,
    label_axis: str = "X(t)",
  ) -> None:
    """
    Plot time-wise mean and percentile band for an array of paths.

    Args:
      t: (N+1,)
      values: (N+1, n_paths), e.g., all x(t) across paths
    """
    mean = values.mean(axis=1)
    lo = np.percentile(values, p_lo, axis=1)
    hi = np.percentile(values, p_hi, axis=1)

    plt.figure(figsize=(8, 4.5))
    plt.plot(t, mean, linewidth=2, label="Ensemble mean")
    plt.fill_between(t, lo, hi, alpha=0.25, label=f"{p_lo}–{p_hi} percentile band")
    plt.xlabel("Time [s]")
    plt.ylabel(label_axis)
    plt.title(f"{label_axis} with Uncertainty Band")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
