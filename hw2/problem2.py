import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import matplotlib.pyplot as plt

    # from solvers import PDIPSolver
    # from qp_tester import PDIPTester

    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Having implemented the primal-dual interior point algorithm for a generic convex quadratic program, we will now apply it to solve the optimal control problem:

    \begin{align}
        \underset{x_\text{0:N}, u_\text{0:N-1}}{\textrm{minimize}} \,\, & x_N^T Q_f x_N + \sum_{k=0}^{N-1} u_k^T R u_k + x_k ^T Q x_k \nonumber \\
        \textrm{subject to} 
        \,\,& x_{k+1} = A_d x_k + B_d u_k, \quad k = 0, \ldots, N-1\nonumber \\ 
        \,\,&  x_0 = x_\text{init} \nonumber\\
        \,\,& x_\text{min} \leq x_k \leq x_\text{max}, \quad k = 1, \ldots, N,\nonumber\\
        \,\,& u_\text{min} \leq u_k \leq u_\text{max}, \quad k = 0, \ldots, N-1,\nonumber
    \end{align}

    where $x_k = (p_k, v_k) \in \mathbb{R}^4$ includes the spacecraft position $p_k \in \mathbb{R}^2$ and velocity $v_k \in \mathbb{R}^2$ as resolve in the inertial frame.
    The spacecraft control $u_k = (u_x, u_y) \in \mathbb{R}^2$ consists of the acceleration applied in the inertial frame.
    Lower and upper bounds are implemented for the state ($x_\text{min}$ and $x_\text{max}$, respectively) and control ($u_\text{min}$ and $u_\text{max}$, respectively) at each time step.
    Finally, the objective function seeks to minimize the control effort expended to drive the system state to the origin.

    Consequently, we use the 2D double integrator dynamics:

    $A_d = \begin{pmatrix} I_{2\times 2} & \Delta h I_{2\times 2}\\ 0_{2\times 2} & I_{2\times 2} \end{pmatrix}$

    $B_d = \begin{pmatrix} \frac{1}{2}\Delta h^2 I_{2\times 2}\\ \Delta h I_{2\times 2} \end{pmatrix}$

    where $\Delta h$ is the discretization time.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In order to leverage the primal-dual interior point solver we implemented, we must construct convert this double integrator problem into standard form:

    \begin{align}
        \underset{x}{\textrm{minimize}} \,\, & \frac{1}{2}x^TQx + q^Tx \nonumber \\
        \textrm{subject to} 
        \,\,& Ax=b\nonumber \\ 
        \,\,&  Gx \leq h \nonumber\\
    \end{align}

    First determine the vector $x \in \mathbb{R}^n$ corresponding to $x_k$ and $u_k$, i.e., you could define $x$ as:

    $x = (x_0, u_0, x_1, u_1, \ldots, u_{N-1}, x_{N-1}, x_N)$

    or

    $x = (x_0, x_1, x_2, \ldots, x_N, u_0, u_1, \ldots, u_{N-1})$.

    Now implement the following methods in the `OCP` class:

    1. `compute_cost_term`:

    2. `compute_LTI_equality_matrix`: use the double integrator $A_h$ and $B_h$ matrices to construct the equality constraint matrix $A$.

    3. `compute_LTI_equality_bc`: construct the equality constraint vector $b$.

    4. `compute_ineq_con`: use the upper/lower bounds for state ($x_\text{min}$ and $x_\text{max}$) and control ($u_\text{min}$ and $u_\text{max}$) to construct the inequality constraint matrix $G$ and vector $h$.
    """
    )
    return


if __name__ == "__main__":
    app.run()
