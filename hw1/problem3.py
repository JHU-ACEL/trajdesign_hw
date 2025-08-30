import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    import marimo as mo

    from gradients import GradientsEval
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Problem 3

    In this problem, we'll evaluate different ways to compute the derivative of a smooth function and leverage the automatic differentiation capability provided by the `jax` library

    Hint: you may find it helpful to scan the [`jax` docs](https://docs.jax.dev/en/latest/automatic-differentiation.html)
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.1

    Consider the multivariable function:

    $f(x, y) = x^2 \cdot y + 3xy + \sin(x) + e^y$

    Start by filling out `gradients.function_eval` to evaluate the above expression.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.2 Compute the derivatives of the function analytically, numerically, and using Jax.

    - First fill out the `analytical_grad` function by computing the gradient by hand
    - Second fill out the `numerical_grad` function by using finite difference approximations of the gradient: $f'(x) = \frac{f(x+\epsilon) - f(x)}{\epsilon}$, where $\epsilon > 0$ is some small number.
    - Finally, fill out the `jax_grad` function and use `jax.gradient` to evaluate the gradient using automatic differentiation.
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
