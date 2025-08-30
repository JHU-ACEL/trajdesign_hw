import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo
    import matplotlib.pyplot as plt

    from solver import Solver
    return jax, jnp, mo, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this problem, we will implement simple versions of gradient descent and Newton's method.

    Start by implementing the following function:
    $$f(x,y) = 0.5(10x^2 + y^2) + 5\log(1+e^{-(x+y)})$$
    """
    )
    return


@app.cell
def _(jnp):
    def function_to_minimize(x: jnp.array) -> float:
        raise NotImplementedError("Function has not been implemented yet")
    return (function_to_minimize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.1 Implement a simple gradient descent solver using a line search approach.

    Fill out the implementation for `solve_with_gradient_descent` and `compute_step_size_ls`

    Note: do not modify the function arguments - we expect that the function will be passed directly as an argument and the gradient computed internally (Hint: use `jax.grad`)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.2 Implement Newton's method for this function

    Fill out the implementation for `solve_with_newton_method`

    Note: as above, do not compute the gradient or Hessian analytically (hint: use `jax.hessian` and `jax.numpy.linalg.solve`)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4.3 Once you're done with both implementations, plot solution trajectory with both methods

    Activate the cell below
    """
    )
    return


@app.cell(disabled=True)
def _(function_to_minimize, jax, jnp, plt):
    xs = jnp.linspace(-10, 10, 100)
    ys = jnp.linspace(-10, 10, 100)

    X, Y = jnp.meshgrid(xs, ys, indexing='ij')  # Shape: (100, 100) each
    coords = jnp.stack([X, Y], axis=-1)  # Shape: (100, 100, 2)

    # First vmap over the second-to-last axis (y-direction)
    vmap_over_y = jax.vmap(function_to_minimize, in_axes=-2)
    # Second vmap over the last remaining spatial axis (x-direction) 
    vmap_over_xy = jax.vmap(vmap_over_y, in_axes=-2)

    results_grid = vmap_over_xy(coords)  # Shape: (100, 100)

    plt.figure(figsize=(8, 6))
    contours = plt.contour(xs, ys, results_grid, levels=20, colors='black', alpha=0.6)
    plt.contourf(xs, ys, results_grid, levels=50, cmap='viridis', alpha=0.8)

    # plt.plot(gradient_soln_traj[:,0], gradient_soln_traj[:,1], color='k')
    # plt.plot(newton_soln_traj[:,0], newton_soln_traj[:,1], color='r')

    plt.colorbar(label='Function value')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Contours')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
