import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this problem, we will be implementing a geometric RRT planner for the same collision avoidance with axis-aligned obstacles task. The pseudo-code for this algorithm can be found in Figure 5 of [this paper](https://msl.cs.uiuc.edu/~lavalle/papers/LavKuf01b.pdf).

    As a part of this, you will fill out the methods in the `RRTSolver` class. Although we don't provide strict guidance on the data structures you use, we suggest that you use the following inside of the `solve` method:
    1. `V`: a data structure that tracks the samples drawn at each iteration of `RRT`.
    2. `P`: a data structure that tracks the parent of each node sampled.
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np

    import matplotlib.pyplot as plt

    from rrtsolver import RRTSolver
    from utils import random_obs
    return RRTSolver, mo, np, random_obs


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 2.1

    Finish implementing the `RRTSolver` class and illustrate a few plots of the planner running for different planning problems.
    """
    )
    return


@app.cell(disabled=True)
def _(np, random_obs):
    n_obs = 16
    posmin = np.array([-5,-5])
    posmax = np.array([5,5])
    obstacles = random_obs(n_obs, posmin, posmax)
    return obstacles, posmax, posmin


@app.cell
def _(RRTSolver, obstacles):
    eps = 0.5
    max_iters = 1000
    goal_bias = 0.05
    rrt = RRTSolver(obstacles, eps, max_iters, goal_bias)
    return (rrt,)


@app.cell
def _(np, posmax, posmin, rrt):
    rrt.set_configuration_space_bounds(posmin, posmax)
    rrt.set_boundary_conditions(np.array([4.5,0]), np.array([-4.5,-2]))
    rrt.solve()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 2.2

    Generate a runtime plot for the `RRT` solver with the x-axis consisting of a different number of obstacles ($N_\text{obs} \in \{ 4, 6, 8, 10, 12 \}$) and the y-axis consisting of the average runtime for different planning problems.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 2.3

    Generate three plots for the same planning problem, but with the maximum number of allowable samples increasing from $100$, $500$ and $1000$ and qualitatively describe your results.
    """
    )
    return


if __name__ == "__main__":
    app.run()
