import re
import numpy as np
from externe_files import file


def give_steps(p0, p1, pillars):
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

        virtual_kv_points = give_steps(p0=_p0, p1=_p1, pillars=n)

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

def is_list(expression):
    if isinstance(expression, list):
        return True
    else:
        return False


def load_bad_pixel_map(crop):
    path_to_map = r'\\132.187.193.8\junk\sgrischagin\BAD-PIXEL-bin1x1-scans-MetRIC_SCAP_IMGS.tif'
    bad_pixel_map = file.image.load(path_to_map)[crop]
    print(f'\n{path_to_map} loaded as bad pixel map. \n')
    return bad_pixel_map


def poly_fit(var_x, var_y, steps):
    def func_poly(x, a, b, c): return a * x ** 2 + b * x + c
    a, b, c = np.polyfit(var_x, var_y, deg=2)
    x = np.linspace(var_x[0], var_x[-1], steps)
    y = func_poly(x, a, b, c)
    return x, y



