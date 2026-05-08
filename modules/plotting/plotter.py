import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from pandas._typing import ArrayLike


class Plotter:
    """
    Sample usage:
        from plotting.plotter import Plotter
        import sympy as sp

        # define function
        plotter = Plotter(-2, 4, (10,6), 500)
        x = plotter.x
        f = x**3 - 3*x**2 + 2
        plotter.plot(f, label='f(x)')
        f_prime = sp.diff(f, x)
        f_integral = sp.integrate(f, x)
        plotter.plot(f_prime, label='f\'(x)')
        plotter.plot(f_integral, label='f_int(x)')
        plotter.show()

    Parameters
    ----------
    figsize: tuple
        Figure size in inches
    start : number
        The starting value of the sequence.
    stop : number
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    """
    def __init__(self, start, stop, figsize: ArrayLike=(10,6), num=50):
        plt.figure(figsize=figsize)
        plt.axhline(0, color='black', linewidth=0.5)
        self._xlinespace = np.linspace(start, stop, num)
        self.x = sp.symbols('x')


    def plot(self,fun, label:str):
        # convert symbolic expressions to numerical functions
        f_num = sp.lambdify(self.x, fun, 'numpy')
        plt.plot(self._xlinespace, f_num(self._xlinespace), label=label)

    def show(self):
        plt.legend()
        plt.grid(True)
        plt.show()
