import numpy as np
from externe_files import file


def load_bad_pixel_map():
    path_to_map = r'\\132.187.193.8\junk\sgrischagin\BAD-PIXEL-bin1x1-scans-MetRIC_SCAP_IMGS.tif'
    bad_pixel_map = file.image.load(path_to_map)
    print(f'bad pixel map from path: {path_to_map} loaded')
    return bad_pixel_map

def poly_fit(var_x, var_y, steps):
    a, b, c = np.polyfit(var_x, var_y, deg=2)
    x = np.linspace(var_x[0], var_x[-1], steps)
    y = func_poly(x, a, b, c)
    return x, y

def func_poly(x, a, b, c):
    return a * x ** 2 + b * x + c

