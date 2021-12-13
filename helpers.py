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


def find_max(array):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = array.argmax()
    return array[idx], idx


def find_min(array):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = array.argmin()
    return array[idx], idx


def find_nearest(array, value):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def merge_v1D(*cols):
    arr = np.vstack(cols).T
    arr.astype(float)
    return arr


def extract_angle(num_of_projections, img_name, num_len):
    """
    :param num_len: num of integers in the naming convention of the image e.g. img_0017
    """

    img_num_str = re.findall(r'[0-9]{4,10}', img_name)[0]

    num = float(img_num_str)
    if num < 1:
        num = 0.0
    else:
        img_num_str = img_num_str.lstrip('0')
        num = int(img_num_str)

    angle = (360/num_of_projections) * num

    return angle

def strip_chars(char, num):
    pass



def find_roots(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)


def line_intersection(line1, line2):
    L1x1 = 0.0
    L1x2 = 180.0
    L1y1 = 0
    L1y2 = 0

    L2x1 = 0
    L2x2 = 0
    L2y1 = 0
    L2y2 = 1

    A = [L1x1, L1y1]
    B = [L1x2, L1y2]
    C = [L2x1, L2y1]
    D = [L2x2, L2y2]

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def remove_duplicates(arr):
    del_rows = []
    for j in range(len(arr[:, 0]) - 1):
        if arr[j, 0] == arr[j + 1, 0]:
            del_rows.append(j)
    del_rows = np.asarray(del_rows)
    arr = np.delete(arr, [del_rows], axis=0)
    return arr


def find_neighbours(map, kV):
    # find first element in map where arg. >= self.U0, which is the right border of the searched interval
    # the nearest left entry is the left border between which the interpolation will take place
    d = None
    for d in map['d_curves']:
        break
    if d is not None:
        _c_kV = map['d_curves'][d]['raw_data'][:, 0].tolist()

        num = next(i[0] for i in enumerate(_c_kV) if i[1] > kV)
        left_nbr = _c_kV[num - 1]
        right_nbr = _c_kV[num]
        dist = kV - left_nbr

        return abs(int(dist)), int(left_nbr), int(right_nbr), (int(right_nbr) - int(left_nbr))
    else:
        print('Could not estimate \'neighbours\'.')


def prep_curve(x, y):
    xd = np.diff(x)
    yd = np.diff(y)
    dist = np.sqrt(xd ** 2 + yd ** 2)
    u = np.cumsum(dist)
    u = np.hstack([[0], u])

    t = np.linspace(0, u.max(), 141)
    xn = np.interp(t, u, x)
    yn = np.interp(t, u, y)
    return xn, yn


#def d_curve_T_intercept(self, curve):
#    idx = None
#    deltas = [0.0, 0.0000001, 0.000001]
#    for delta in deltas:
#        nearest_val, idx = h.find_nearest(curve[:, 0], self.T_min + delta)
#        if idx is not None:
#            self.intercept_found = True
#            break
#    self.intercept['x'] = self.T_min
#    self.intercept['y'] = curve[:, 1][idx]