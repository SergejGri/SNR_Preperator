import re
import numpy as np
from externe_files import file



def extract_d(dfile):
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
    bad_pixel_map = file.image.load(path_to_map)[crop]
    print(f'\n{path_to_map} loaded as bad pixel map. \n')
    return bad_pixel_map

def poly_fit(var_x, var_y, steps):
    a, b, c = np.polyfit(var_x, var_y, deg=2)
    x = np.linspace(var_x[0], var_x[-1], steps)
    y = func_poly(x, a, b, c)
    return x, y

def func_poly(x, a, b, c):
    return a * x ** 2 + b * x + c

