import re
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates as sci

from ext import file


def give_lin_steps(p0: list, p1: list, pillars):
    """
    Estimates a evenly spaced divided length between two points via pythagoras
    :param p0:  should have the form [x0, y0]. Where x0 is the x value of your first data point and y0 the y value
    :param p1:  See param p0.
    :param pillars: number of steps you want to divide you length
    """
    if isinstance(pillars, int):
        return np.vstack([np.linspace(p0[0], p1[0], pillars), np.linspace(p0[1], p1[1], pillars)]).T


def create_kv_grid(data_curve):
    _c = data_curve
    kv_grid = np.empty(shape=3)

    for i in range(len(_c[:, 0])-1):
        _p0 = []
        _p1 = []

        # TODO: change the selection of kvs depending on the curve
        kv0 = int(_c[i, 0])
        kv1 = int(_c[i+1, 0])
        n = abs(int(kv1) - int(kv0) + 1)

        row0 = _c[_c[:, 0] == kv0]
        row1 = _c[_c[:, 0] == kv1]

        _p0 = [row0[:, 1][0], row0[:, 2][0]]
        _p1 = [row1[:, 1][0], row1[:, 2][0]]

        virtual_kv_points = give_lin_steps(p0=_p0, p1=_p1, pillars=n)

        kvs_vals = np.linspace(kv0, kv1, n)[np.newaxis].T
        kvs_vals = np.hstack((kvs_vals, virtual_kv_points))
        kv_grid = np.vstack((kv_grid, kvs_vals))

    # fine tune new curve
    kv_grid = kv_grid[1:]
    del_rows = []
    for j in range(len(kv_grid[:, 0])-1):
        if kv_grid[j, 0] == kv_grid[j+1, 0]:
            del_rows.append(j)
    del_rows = np.asarray(del_rows)
    kv_grid = np.delete(kv_grid, [del_rows], axis=0)
    return kv_grid


def is_int(var):
    if var % 1.0 == 0.0:
        return True
    else:
        return False


def extract_d(dfile):
    d_str = re.findall("([0-9]+)mm", dfile)
    if len(d_str) < 1:
        d_str = re.findall("([0-9]+)_mm", dfile)
    if is_list(d_str):
        return int(d_str[0])
    else:
        return int(d_str)


def extract_kv(dfile):
    d_str = re.findall("([0-9]+)kV", dfile)
    if len(d_str) < 1:
        d_str = re.findall("([0-9]+)_kV", dfile)
    if is_list(d_str):
        return int(d_str[0])
    else:
        return int(d_str)


def extract(what: str, dfile: str):
    '''
    :param param:   Two cases are possible: param='d' or param='kv'
    :param dfile:   must be a path or file name with a naming convention which contains units for extraction.
    '''

    d_str = ''
    if what == 'kv':
        d_str = re.findall("([0-9]+)kV", dfile)
        if len(d_str) < 1:
            d_str = re.findall("([0-9]+)_kV", dfile)
        if is_list(d_str):
            return int(d_str[0])
        else:
            return int(d_str)
    elif what == 'd':
        d_str = re.findall("([0-9]+)mm", dfile)
        if len(d_str) < 1:
            d_str = re.findall("([0-9]+)_mm", dfile)
        if is_list(d_str):
            return int(d_str[0])
        else:
            return int(d_str)



def is_list(expression):
    if isinstance(expression, list):
        return True
    else:
        return False


def load_bad_pixel_map(crop):
    path_to_map = r'\\132.187.193.8\junk\sgrischagin\BAD-PIXEL-bin1x1-scans-MetRIC_SCAP_IMGS.tif'
    _crop = slice(crop[0][0], crop[0][1]), slice(crop[1][0], crop[1][1])
    bad_pixel_map = file.image.load(path_to_map)[_crop]
    print(f'\n{path_to_map} loaded as bad pixel map. \n')
    return bad_pixel_map

def linear_f(x, m, t):
    return m*x + t

def poly_2(x, a, b, c):
    return a * x ** 2 + b * x + c


def poly_fit(var_x, var_y, steps):
    def func_poly(x, a, b, c): return a * x ** 2 + b * x + c
    a, b, c = np.polyfit(var_x, var_y, deg=2)
    x = np.linspace(var_x[0], var_x[-1], steps)
    y = func_poly(x, a, b, c)
    return x, y


def test_plot():
    x = np.array([0.4, 0.45, 0.50, 0.55, 0.6, 0.63, 0.65, 0.7, 0.73, 0.75])
    y = np.array([1.2, 1.26, 1.3, 1.32, 1.325, 1.324, 1.32, 1.28, 1.25, 1.21])
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)

    xp = xp = np.linspace(x[0], x[-1], 1000)
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.plot(xp, p(xp), label='fit')
    plt.scatter(0.47, p(0.47), c='red')
    plt.legend()
    plt.show()