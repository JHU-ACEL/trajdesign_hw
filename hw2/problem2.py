import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import jax.scipy as jsp
    import matplotlib.pyplot as plt

    from solvers import PDIPSolver
    from ocp import OCPSolver
    from qp_tester import PDIPTester

    import marimo as mo
    return OCPSolver, PDIPSolver, jnp, mo


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Now test out your implementation. The following has a simple set of parameters for the number of time steps $N$ and and time step $\Delta h$.""")
    return


@app.cell
def _(jnp):
    nx = 4
    nu = 2
    N = 1

    dh = 0.5
    Ak = jnp.vstack((
        jnp.hstack((jnp.eye(2), dh*jnp.eye(2))),
        jnp.hstack((jnp.zeros((2,2)), jnp.eye(2)))
    ))
    Bk = jnp.vstack((
        0.5*dh**2*jnp.eye(2),
        dh*jnp.eye(2)
    ))
    return Ak, Bk, N, nu, nx


@app.cell
def _(OCPSolver):
    ocp_solver = OCPSolver()
    return (ocp_solver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Define $x_\text{init}$ as the initial position and velocity of the spacecraft in 2D""")
    return


@app.cell
def _(jnp):
    x_init = jnp.array([9.4, 7.2, -0.5, 1.0])
    return (x_init,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Define cost matrices $Q_\text{mat}$ and $R_\text{mat}$.

    **Note** that these matrices are not the same as the $Q$ and $R$ matrices above! These $Q_\text{mat}$ and $R_\text{mat}$ are the matrices used to compute stagewise cost, i.e., $x_k^T Q_\text{math} x_k$.
    """
    )
    return


@app.cell
def _(jnp, nu, nx):
    Q_mat = 2*jnp.eye(nx)
    R_mat = jnp.eye(nu)
    return Q_mat, R_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Instantiate upper and lower bounds for state and control""")
    return


@app.cell
def _(jnp):
    x_min = jnp.array([-10.0, -10, -5, -5])
    x_max = jnp.array([10.0, 10, 5, 5])

    u_min = jnp.array([-15.0, -15])
    u_max = jnp.array([15.0, 15])
    return u_max, u_min, x_max, x_min


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Once you've defined all of the methods in the `OCPSolver` class, activate the cell below and instantiate the QP corresponding to the optimal control problem at the top of the page.""")
    return


@app.cell(disabled=True)
def _(
    Ak,
    Bk,
    N,
    Q_mat,
    R_mat,
    jnp,
    nu,
    nx,
    ocp_solver,
    u_max,
    u_min,
    x_init,
    x_max,
    x_min,
):
    A_eq = ocp_solver.compute_equality_matrix(Ak, Bk, N)
    b_eq = ocp_solver.compute_equality_bc(x_init, Ak, N)

    G_ineq, h_ineq = ocp_solver.compute_ineq_con(x_min, x_max, u_min, u_max, N)

    P = ocp_solver.compute_cost_term(Q_mat, R_mat, N)
    p = jnp.zeros(N*(nx+nu))
    return A_eq, G_ineq, P, b_eq, h_ineq, p


@app.cell(disabled=True)
def _(A_eq, G_ineq, P, PDIPSolver, b_eq, h_ineq, p):
    solver = PDIPSolver()
    solver.init_problem(2*P, p, A_eq, b_eq, G_ineq, h_ineq)
    costs = solver.solve_qp(verbose=True)
    return


if __name__ == "__main__":
    app.run()
