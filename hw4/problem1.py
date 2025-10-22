import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this problem, we will be implementing a mixed-integer linear programming (MILP) formulation for collision avoidance in the presence of axis-aligned obstacles:

    \begin{align}
        \underset{x_{0:N}, u_{0:N-1}, \delta_{0:N}}{\textrm{minimize}} \,\, & \sum_{k=1}^{N-1} (x_k-x_g)^T Q (x_k-x_g) + u_k^T R u_k  \nonumber \\
        \textrm{subject to} 
        \,\,& x_0 = x_\text{init}\nonumber \\ 
        \,\,&  x_{k+1} = A_k x_k + B_k u_k, \quad k = 0, \ldots, N-1 \nonumber\\
        \,\,&  x_\text{min} \leq x_k \leq x_\text{max}, \quad k = 0, \ldots, N \nonumber\\
        \,\,&  \| u_k \|_2 \leq u_\text{max}, \quad k = 0, \ldots, N-1 \nonumber\\
        \,\,&  x_k \notin \mathcal{X}_\text{obs} \nonumber\\.
    \end{align}

    where $x_k \in \mathbb{R}^{n_x}$, $u_k \in \mathbb{R}^{n_u}$, and $\delta_k \in \{ 0, 1 \}^{n_z}$. 
    The cost matrices $Q \in \mathbf{S}^{n_x}_+$ and $R \in \mathbf{S}^{n_u}_+$.
    Finally, $\mathcal{X}_\text{obs}$ is given by a set of $N_\text{obs}$ obstacles $\{o_1,\ldots, o_{N_\text{obs}} \}$ where $\mathcal{o}_i = (o_{i, x_\text{min}}, o_{i, x_\text{max}}, o_{i, y_\text{min}}, o_{i, y_\text{max}})$.

    As a part of this, you will fill out the methods in the `MICPSolver` class:

    1. `__init__`: the class constructor contains all of the parameters you'll need to define the constraints.
    2. `construct_dynamics_matrices`: fill out the expressions for the discretized double integrator matrices $A_k$ and $B_k$.
    3. `construct_obs`: use the helper functions from `utils.py` to create a random set of obstacles.
    4. `init_bin_problem`: construct the `cvxpy` problem instance associated with the collision avoidance problem.
    5. `solve_micp`: solve an instance of the collision avoidance problem for a given set of problem parameters.


    A few further notes:

    - We will be constructing a parametric `cvxpy` problem using `cp.Parameter`. This allows us to create a common `cvxpy` problem instance and solve it repeatedly for various parameters.
    """
    )
    return


@app.cell
def _():
    import os
    import yaml
    import pickle
    import sys
    import pdb

    import cvxpy as cp
    import numpy as np
    import matplotlib.pyplot as plt
    import marimo as mo

    from micpsolver import MICPSolver
    from utils import random_obs, findIC
    return MICPSolver, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 1.1

    Fill out the implementation for the necessary class methods and plot a 2 - 3 trajectories that demonstrate the collision-avoidance behavior with the obstacles plotted. Use `plt.scatter` and `plt.quiver` to illustrate the spacecraft velocity along the trajectory as well.
    """
    )
    return


@app.cell(disabled=True)
def _(MICPSolver, np):
    Q = np.diag([50.0,50.0,10,10.])
    R = 10.*np.eye(2)

    dh = 0.5
    N = 11
    n_obs = 8

    solver = MICPSolver(N, dh, n_obs, Q, R)
    solver.init_bin_problem()
    return (solver,)


@app.cell
def _(solver):
    x0, obstacles = solver.construct_obs()
    return obstacles, x0


@app.cell
def _(np, obstacles, solver, x0):
    p_dict = {}
    p_dict['obstacles'] = np.reshape(np.concatenate(obstacles, axis=0), (solver.n_obs,4)).T
    p_dict['x0'] = x0
    p_dict['xg'] = np.array([0.5, 2.0, 0.0, 0.0])

    prob_success, cost, solve_time, (x_star, u_star, y_star) = solver.solve_micp(p_dict)
    print(f'Solver succeeded in {solve_time} [s]!') if prob_success else print('Solver failed!')
    return p_dict, prob_success, x_star


@app.cell
def _(obstacles, p_dict, plt, prob_success, solver, x0, x_star):
    plt.axes()
    for obs in obstacles:
        rect = plt.Rectangle((obs[0], obs[2]), \
                                  obs[1]-obs[0], obs[3]-obs[2], \
                                 fc='red', ec='blue')
        plt.gca().add_patch(rect)
        plt.axis('scaled')


    plt.gca().add_patch(plt.Circle((x0[0],x0[1]), 0.04, fc='blue',ec="green"))
    plt.gca().add_patch(plt.Circle((p_dict['xg'][0],p_dict['xg'][1]), 0.04, fc='green',ec="green"))

    if prob_success:
        plt.quiver(x_star[0,:], x_star[1,:], x_star[2,:], x_star[3,:])
        plt.scatter(x_star[0,:], x_star[1,:])

    ax_ = plt.gca()
    ax_.margins(0)
    ax_.set(xlim=(solver.posmin[0],solver.posmax[0]), ylim=(solver.posmin[1],solver.posmax[1]))
    fig_ = plt.figure()

    plt.show();
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 1.2

    Illustrate how the runtime scales with an increasing horizon $N$ and number of obstacles $N_\text{obs}$. Draw a line plot with an increasing horizon $N$ on the x-axis (say $N \in \{ 10, 20, 30, 40 \}$) and the solution time for ten $feasible$ different random instances for each problem. Generate this plot for $N_\text{obs} \in \{ 4, 6, 8, 10 \}$.
    """
    )
    return


if __name__ == "__main__":
    app.run()
