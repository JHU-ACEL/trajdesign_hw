import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    from jax import lax, jit
    import jax.numpy as jnp
    import numpy as np
    import marimo as mo
    import matplotlib.pyplot as plt

    from propagation2d import Propagation2D
    return Propagation2D, jnp, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Problem 2""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Consider a simple 2D system of a point moving on a plane. Your state consists of its position (p_x and p_y) and velocity (v_x and v_y) in 2D and your input are the respective accelerations (a_x and a_y)

    Consider the following system:

    $x(t)=[p_x,p_y,v_x,v_y]$     (p_x and p_y are 2D position and v_x and v_y are 2D velocities at any instant 't')

    $u(t)=[a_x,a_y]$      (a_x and a_y are accelleration at time 't' and are your control inputs)

    $\dot{x}(t) = [\dot{p_x},\dot{p_y},\dot{v_x},\dot{v_y}] = [v_x,v_y,a_x,a_y]$

    Hence, the continous time dynamics can be represented as:

    $\dot{x}(t) = Ax(t) + Bu(t)$

    where,

    \[
    A = \begin{bmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
    \end{bmatrix},
    \quad
    B = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 1
    \end{bmatrix}
    \]
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 2.1 Propagate the system using matrix exponential. Store the state of the system at each time step and return the final state of the system. Plot the trajectory (x and y positon) using the states array.""")
    return


@app.cell
def _(jnp):
    A =  jnp.array([])    # Fill out A matrix
    B = jnp.array([])     # Fill out B matrix
    return A, B


@app.cell
def _(A, B, Propagation2D):
    prop = Propagation2D(A, B)

    #Define initial conditions for the simulations
    dt = 0.1 #step size in seconds
    total_time = 5 #Simulation time in seconds

    # Define initial conditions
    state_0 = prop.state_t(1, 1, 1, 1)   # [x, y, vx, vy]
    input_0 = prop.input_t(2, 1)         # [ax, ay]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Fill out the expressions for `c2d_double_integrator_closed_form` and `analytical_solution` in `propagation2d` and then plot the ensuing trajectory using `plot_traj`""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.2 - Repeat the same as above using forward Euler and RK4.

    Fill out `euler_step`, `euler_normal`, `rk4_step`, and `rk4_normal`
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.3 - Compare the final state of the system using Euler and RK4 with matrix exponential and return the final x and y error.


    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 2.4 - Now consider your control inputs (in the above case acceleration) has white noise (process noise) which gets injected into the system. The new system would then no longer be a deterministic system but rather a stochastic one. Propagate this new stochastic system using the Euler Maruyama method and plot different possible trajectories using the same initial state and control inputs as above. Also show the mean-percentile graph of both x and y position.

    Fill out `em_step` and `simulate_em` to implement the Euler-Maruyama version of this problem and use the `plot_mean_with_percentile_band` helper function
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Euler–Maruyama Method (SDEs)
    In general
    For the SDE:

    \[
    dX_t = a(X_t, t)\,dt + b(X_t, t)\,dW_t
    \]

    the Euler–Maruyama discretization is:

    \[
    X_{n+1} = X_n + a(X_n, t_n)\,\Delta t + b(X_n, t_n)\,\Delta W_n
    \]

    where:

    - \(\Delta t\) is the time step,  
    - \(\Delta W_n = \sqrt{\Delta t}\, Z_n,\quad Z_n \sim \mathcal{N}(0,1)\).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Stochastic 2D Double Integrator

    The stochastic 2D double integrator in SDE form is:

    \begin{align}
    d x_t &= v_x \, dt, \\
    d y_t &= v_y \, dt, \\
    d v_{x,t} &= a_x(x,y,v_x,v_y,t)\, dt + \sigma_x \, dW_{x,t}, \\
    d v_{y,t} &= a_y(x,y,v_x,v_y,t)\, dt + \sigma_y \, dW_{y,t}.
    \end{align}

    where \(W_{x,t}, W_{y,t}\) are independent Wiener processes and 
    \(\sigma_x, \sigma_y \geq 0\) set the noise strength.


    In compact **matrix form**:

    \[
    dX_t = (A X_t + B u_t)\, dt + G\, dW_t,
    \]

    where

    \[
    X_t = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}, \quad
    u_t = \begin{bmatrix} a_x \\ a_y \end{bmatrix},
    \]

    \[
    A = \begin{bmatrix}
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    0 & 0 & 0 & 0 \\
    0 & 0 & 0 & 0
    \end{bmatrix}, \quad
    B = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    1 & 0 \\
    0 & 1
    \end{bmatrix}, \quad
    G = \begin{bmatrix}
    0 & 0 \\
    0 & 0 \\
    \sigma_x & 0 \\
    0 & \sigma_y
    \end{bmatrix}, \quad
    dW_t = \begin{bmatrix} dW_{x,t} \\ dW_{y,t} \end{bmatrix}.
    \]

    With step size \(\Delta t\) and \(\Delta W \sim \mathcal{N}(0,\Delta t)\):

    \[
    \begin{aligned}
    x_{k+1}   &= x_k   + v_{x,k}\,\Delta t, \\
    y_{k+1}   &= y_k   + v_{y,k}\,\Delta t, \\
    v_{x,k+1} &= v_{x,k} + a_x(\cdot)\,\Delta t + \sigma_x \sqrt{\Delta t}\,\xi_{x,k}, \\
    v_{y,k+1} &= v_{y,k} + a_y(\cdot)\,\Delta t + \sigma_y \sqrt{\Delta t}\,\xi_{y,k},
    \end{aligned}
    \]

    where \(\xi_{x,k}, \xi_{y,k} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)\).
    """
    )
    return


if __name__ == "__main__":
    app.run()
