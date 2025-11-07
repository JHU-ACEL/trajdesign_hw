import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Homework Problem: MPPI and Cross Entropy for Trajectory Generation (Dynamic Bicycle Model)

    In this problem, you will implement sampling-based trajectory optimization: **Model Predictive Path Integral (MPPI)** and **Cross Entropy Method (CEM)** for a nonlinear system, specifically the **dynamic bicycle model**.

    ## Dynamics

    The **dynamic bicycle model** describes car-like robot motion as:

    - **State:** $\mathbf{x} = [x,\, y,\, \theta,\, v,\, \delta]$
      - $x, y$: position,
      - $\theta$: heading,
      - $v$: velocity,
      - $\delta$: steering angle.

    - **Control:** $\mathbf{u} = [a,\, \dot{\delta}]$
      - $a$: acceleration,
      - $\dot{\delta}$: steering rate.

    Discrete updates (Euler):


    **Discrete-time update equations:**

    $$
    \begin{aligned}
    x_{t+1}      &= x_t + v_t \cos(\theta_t) \Delta t \\
    y_{t+1}      &= y_t + v_t \sin(\theta_t) \Delta t \\
    \theta_{t+1} &= \theta_t + \frac{v_t}{L} \tan(\delta_t) \Delta t \\
    v_{t+1}      &= v_t + a_t \Delta t \\
    \delta_{t+1} &= \delta_t + \dot{\delta}_t \Delta t
    \end{aligned}
    $$

    ---

    ## Optimization Methods

    ### Model Predictive Path Integral (MPPI)

    MPPI is a **sampling-based stochastic optimal control method**:

    - At each iteration, **sample** $K$ control sequences $\{\mathbf{u}^{(k)}\}_{k=1}^K$ by adding noise to a nominal control sequence.
    - **Roll out** each trajectory according to the system dynamics.
    - **Compute cost** $J_k$ for each sample.
    - **Weight** each control trajectory by its (exponentiated negative) cost:

      $w_k = \frac{\exp\left(-\frac{1}{\kappa} J_k\right)}{\sum_{i=1}^K \exp\left(-\frac{1}{\kappa} J_i\right)}$

      where $\kappa$ is a temperature tuning parameter.

    - **Update the nominal control sequence** by weighted average:

      $\mathbf{u}_t^{\text{new}} = \sum_{k=1}^K w_k\, \mathbf{u}_t^{(k)}$

    - Apply or rollout with the updated controls.

    ---

    ### Cross Entropy Method (CEM)

    CEM optimizes the control sequence distribution by **focusing sampling on elite solutions**:

    - At each iteration, **sample** $K$ control sequences $\{\mathbf{u}^{(k)}\}$ from the current distribution ($\mathcal{N}(\mu,\, \Sigma)$).
    - **Roll out** and **evaluate cost** for each sample.
    - **Select the best (elite) $K_e$ samples** with lowest costs.
    - **Update the distribution parameters**:
      - New mean $\mu_{\text{new}}$ = mean of elites,
      - New std $\sigma_{\text{new}}$ = std of elites.
    - Repeat for a number of iterations.

    ---

    ## Assignment

    - Implement the dynamic bicycle model step.
    - Implement MPPI and CEM.
    - Define a quadratic cost:
    $J = (\mathbf{x}_N - \mathbf{x}_{goal})^T Q_f (\mathbf{x}_N - \mathbf{x}_{goal}) + \sum_{t=0}^{N-1}  ((\mathbf{x}_t - \mathbf{x}_{goal})^T Q (\mathbf{x}_t - \mathbf{x}_{goal}) + \mathbf{u}_t^T R \mathbf{u}_t)$
    - **Compare** trajectories produced by both optimizers, especially as you vary the number of samples, iterations, and cost weights.
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    from mppi_cem import MppiCemSolver
    return MppiCemSolver, mo, np, plt


@app.cell
def _(MppiCemSolver, np):
    # Problem parameters
    dt = 0.2
    T = 30    # time horizon (steps)
    x0 = np.array([0., 0., 0., 2., 0.])         # [x, y, theta, v, delta]
    x_goal = np.array([10., 6., 0.0, 0., 0.])    # target at (15, 8), v=2, theta=0, delta=0

    solver = MppiCemSolver(
        x0=x0, x_goal=x_goal, dt=0.1, T=T,
        num_samples=200, num_iter=15, kappa=2.0
    )
    return solver, x0, x_goal


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 1. MPPI: Run the solver and plot trajectory

    Run `solve_mppi()` below. Plot the trajectory in $(x, y)$ and the controls over time.
    """
    )
    return


@app.cell
def _(solver):
    states, controls = solver.solve_mppi()
    return controls, states


@app.cell
def _(plt, states, x0, x_goal):
    plt.figure(figsize=(6, 4))
    plt.plot(states[:, 0], states[:, 1], 'b-', label='Trajectory')
    plt.scatter([x0[0], x_goal[0]], [x0[1], x_goal[1]], c=['g','r'], label='Start/Goal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("MPPI: State Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(controls, plt):
    plt.figure(figsize=(6, 3))
    plt.plot(controls[:,0], label="acc")
    plt.plot(controls[:,1], label="steering rate")
    plt.xlabel("Timestep")
    plt.ylabel("Input")
    plt.title("MPPI: Control Inputs")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2. CEM: Run the solver and plot trajectory

    Run `solve_cem()` below. Plot the trajectory in $(x, y)$ and the controls over time.
    """
    )
    return


@app.cell
def _(solver):
    states_cem, controls_cem = solver.solve_cem()
    print("CEM Trajectory shape:", states_cem.shape)
    print("CEM Controls shape:", controls_cem.shape)
    return controls_cem, states_cem


@app.cell
def _(plt, states_cem, x0, x_goal):
    plt.figure(figsize=(6, 4))
    plt.plot(states_cem[:, 0], states_cem[:, 1], 'm-', label='CEM Trajectory')
    plt.scatter([x0[0], x_goal[0]], [x0[1], x_goal[1]], c=['g','r'], label='Start/Goal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("CEM: State Trajectory")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell
def _(controls_cem, plt):
    plt.figure(figsize=(6, 3))
    plt.plot(controls_cem[:,0], label="acc")
    plt.plot(controls_cem[:,1], label="steering rate")
    plt.xlabel("Timestep")
    plt.ylabel("Input")
    plt.title("CEM: Control Inputs")
    plt.legend()
    plt.grid()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Plot trajectories for both MPPI and CEM and varying key parameters
           (cost function, time horizon, number of iterations, etc.). 
          """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
