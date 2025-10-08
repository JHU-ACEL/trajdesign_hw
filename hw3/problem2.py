import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this problem, we will implement an optimal controller for rotational maneuvers for the `Astrobee` robot [[1](https://ieeexplore.ieee.org/document/7759803/similar#similar)] using `acados`.

    The quaternion convention we will use is: $q = (q_s, q_v)$ where $q_s$ and $q_v$ are the scalar and vector components, respectively.
    The optimal control problem consists of:

    \begin{align}
        \underset{q_{0:N}, \omega_{0:N}, M_{0:N}}{\textrm{minimize}} \,\, & \sum_{k=0}^N (1-\|q_k^Tq_g\|^2)_Q + M_k^TRM_k \nonumber \\
        \textrm{subject to} 
        \,\,& q_{k+1} = f_q(q_k, \omega_k) \\
        \,\,& \omega_{k+1} = f_\omega(q_k, \omega_k, M_k) \\
        \,\,&  \omega_\text{min} \leq \omega_k \leq \omega_\text{max}\\
        \,\,&  M_\text{min} \leq M_k \leq M_\text{max},
    \end{align}

    where $f_q(q_k, \omega_k)$ is the discretized form of the quaternion kinematics $\dot{q} = \frac{1}{2}\Omega(\omega)q$ and $f_\omega(\cdot)$ is the discretized form of the angular dynamics $\dot{\omega} = J^{-1}(M - \omega \times J \omega)$, where $J$ is the inertia tensor.
    The cost function penalizes angular distance to some reference orientation $q_g$ and control effort.

    For the `AstrobeeImplicitModel` class, you'll have to implement the following methods:

    1. `quaternion_kinematic_matrix`: implement the $\Omega(\omega)$ matrix associated with quaternion kinematics $\dot{q} = \frac{1}{2}\Omega(\omega)q$. Recall that $\Omega(\omega) = \begin{pmatrix}0 & -\omega^T \\ -\omega & -\omega^\times\end{pmatrix}$.
    2. `get_acados_model`: construct the `AcadosModel` for rotation planning.

    For the `AstrobeeMPC` class, you'll have to implement the following methods:
    1. `setup`: This method setups the `AcadosOcpSolver` object and explicitly defines the cost terms and constraints.
    2. `solve`: This method solves the optimal control problem given an initial state `x0`.

    Suggestions:

    - Take a look at some of the `Python` examples provided [here](https://github.com/acados/acados/tree/main/examples/acados_python).
    - The [problem formulation document](https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf) will also be a useful guide to specify the necessary parameters for the `AcadosOcpSolver`.
    - For the cost term, I suggest using the `NONLINEAR_LS` option as it allows for defining the orientation cost.
    - For the system dynamics, I suggest implementing a discrete integrator of your choice (e.g., forward Euler) and then normalizing the quaternion to enforce the unit norm constraint. This can be managed by setting the `integrator_type` to `DISCRETE`.
    """
    )
    return


@app.cell
def _():
    from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver, AcadosModel
    import numpy as np
    import casadi as cs
    import os
    import matplotlib.pyplot as plt

    import marimo as mo

    from astrobee_model import AstrobeeImplicitModel
    from astrobee_mpc import AstrobeeMPC
    return AstrobeeImplicitModel, AstrobeeMPC, mo, np, plt


@app.cell(disabled=True)
def _(AstrobeeImplicitModel):
    astrobee_model = AstrobeeImplicitModel()
    return (astrobee_model,)


@app.cell
def _(AstrobeeMPC, astrobee_model):
    astrobee_mpc = AstrobeeMPC(astrobee_model)
    return (astrobee_mpc,)


@app.cell
def _(astrobee_mpc, np):
    x0 = np.array([0.5, -0.5, 0.5, 0.5, 0, 0, 0])
    simX, simU = astrobee_mpc.solve(x0)
    return simU, simX


@app.cell
def _(plt, simX):
    plt.plot(simX[:, 0])
    plt.plot(simX[:, 1], linestyle='--')
    plt.plot(simX[:, 2], linestyle='-.')
    plt.plot(simX[:, 3], linestyle='--', color='k')
    return


@app.cell
def _(plt, simU):
    plt.plot(simU[:,0])
    plt.plot(simU[:,1], linestyle='--')
    plt.plot(simU[:,2], linestyle='--')
    return


if __name__ == "__main__":
    app.run()
