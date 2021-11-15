import numpy as np


def poly_fit(var_x, var_y, steps):
    a, b, c = np.polyfit(var_x, var_y, deg=2)
    x = np.linspace(var_x[0], var_x[-1], steps)
    y = func_poly(x, a, b, c)
    return x, y

def func_poly(x, a, b, c):
    return a * x ** 2 + b * x + c

