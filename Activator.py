import os
import gc
import csv
import sys

import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

from ext import file
from visual.Plotter import Plotter as PLT
from visual.Plotter import TerminalColor as PCOL
from snr_calc.map_generator import SNRMapGenerator
from snr_calc.preperator import ImageLoader
from snr_calc.preperator import calc_T
import helpers as h


def fast_CT(num_proj):
    crop = (500, 1500), (595, 1395)

    p_imgs = r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\fast_CT'
    p_darks = r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\darks'
    p_refs = r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\refs'

    img_holder = ImageLoader(used_SCAP=False, remove_lines=True, load_px_map=False)
    data = img_holder.load_stack(path=p_imgs)
    darks = img_holder.load_stack(path=p_darks)
    refs = img_holder.load_stack(path=p_refs)

    list_T = []
    list_angles = []
    all_imgs = [f for f in os.listdir(p_imgs) if os.path.isfile(os.path.join(p_imgs, f))]
    for i in range(data.shape[0]):
        theta = h.extract_angle(num_of_projections=num_proj, img_name=all_imgs[i], num_len=4)

        T = calc_T(data=data, refs=refs, darks=darks)
        list_T.append(T)
        list_angles.append(theta)

    T = np.asarray(list_T)
    theta = np.asarray(list_angles)
    del data, refs, darks
    gc.collect()

    CT_data = h.merge_v1D(theta, T)
    return CT_data


class Scanner:
    def __init__(self, snr_files: str, T_files: str, ds_ex: list):
        self.p_SNR_files = snr_files
        self.p_T_files = T_files
        self.ds_ex = ds_ex
        self.path_fin = os.path.join(os.path.dirname(self.p_T_files), 'MAP')
        self.curves = {}
        self.files = {}

        self.collect_snr_files()
        self.collect_transmission_files()

    def collect_snr_files(self):
        loc = []
        for _dir in os.listdir(self.p_SNR_files):
            _subdir = os.path.join(self.p_SNR_files, _dir)
            for file in os.listdir(_subdir):
                if file.endswith('.txt'):
                    d = h.extract(what='d', dfile=file)
                    if d in self.ds_ex:
                        pass
                    else:
                        loc.append(os.path.join(_subdir, file))
        self.files['SNR'] = loc

    def collect_transmission_files(self):
        loc_fs = []
        loc_ds = []
        for file in os.listdir(self.p_T_files):
            if file.endswith('.csv'):
                d = h.extract(what='d', dfile=file)
                if d in self.ds_ex:
                    pass
                else:
                    loc_ds.append(d)
                    loc_fs.append(os.path.join(self.p_T_files, file))
        self.files['T'] = loc_fs
        loc_ds = sorted(loc_ds, key=lambda x: int(x))
        self.files['ds'] = loc_ds

    def collect_curve_data(self, d):
        loc_list = []
        for file in os.listdir(self.p_curves):
            if file.endswith('.csv') and f'{d}_mm' in file:
                loc_list.append(os.path.join(self.p_curves, file))
        return loc_list

    @staticmethod
    def extract_values(file):
        kV = []
        T = []
        SNR = []
        with open(file) as fp:
            reader = csv.reader(fp, delimiter=',')
            data_read = [row for row in reader]
        for i in range(len(data_read)):
            kV.append(float(data_read[i][0]))
            T.append(float(data_read[i][1]))
            SNR.append(float(data_read[i][2]))
        return kV, T, SNR


class Activator:
    def __init__(self, snr_files: str, T_files: str, U0: int, snr_user: float, kv_ex: list = None, ds_ex: list = None,
                 ssize=None, vir_curve_step: float = None, create_plot: bool = False):

        self.fast_CT_data = None
        self.T_min = None

        self.kv_ex = kv_ex
        self.ds_ex = ds_ex

        if ssize:
            self.ssize = ssize
        else:
            self.ssize = (250, 150)
            self.init_MAP = True
        self.snr_user = snr_user
        self.scanner = Scanner(snr_files=snr_files, T_files=T_files, ds_ex=self.ds_ex)

        self.stop_exe = False

        if 40 <= U0 <= 180:
            self.U0 = U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {U0} \n'
                  + PCOL.BOLD + '...exit...' + PCOL.END)
            sys.exit()

        self.kV_interpolation = False

        if vir_curve_step is None:
            self.vir_curve_step = 0.1
        else:
            self.vir_curve_step = vir_curve_step

        self.U0_intercept = {'x': {}, 'y': {}, 'd': {}}

        self.Ubest_curve = {'val': None, 'fit': {}, 'data': {}}
        self.U0_curve = {'val': self.U0, 'fit': {}, 'raw_data': {}}

        self.Generator = SNRMapGenerator(scanner=self.scanner, kv_filter=kv_ex)

    def __call__(self, create_plot: bool = True, detailed: bool = False):
        # self.fast_CT_data = fast_CT()
        # 0) find U_best
        # 1) fast_CT
        # 2) extract T_min
        # 3)
        self.fast_CT_data = fast_CT(num_proj=1500)
        self.fast_CT_data = [[0.513, 0.157, 0.319, 0.419, 0.351, 0.359, 0.473], [0.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]]
        self.T_min, _ = h.find_min(self.fast_CT_data[0])

        self.map = self.Generator(spatial_range=self.ssize)
        self.map['T_min'] = self.T_min
        self.map['iU0'] = {'x': None, 'y': None, 'd': None}

        self.create_U0_curve(U0=self.U0)

        self.map['iU0']['x'], \
        self.map['iU0']['y'], \
        self.map['iU0']['d'] = self.find_intercept(kv_val=self.U0, T_val=self.T_min)

        if self.map['iU0']['y'] is not None:
            self.create_Ubest_curve(d=self.map['iU0']['d'])
            self.printer()

        if create_plot:
            _plt = PLT()
            _plt.create_T_kv_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)
            _plt.create_MAP_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)

        self.create_lookup_table()
        # self.calc_t_exp()

    def get_min_T(self):
        return np.min(self.fast_CT_data[0])

    def vertical_interpolation(self, points: np.array):
        step = 100000
        _f = interpolate.interp1d(points[:, 0], points[:, 1], kind='linear')
        x_fit = np.linspace(points[:, 0][0], points[:, 0][-1], step)
        y_fit = _f(x_fit)
        data = np.vstack((_f.x, _f.y)).T
        curve = np.vstack((x_fit, y_fit)).T
        return curve, data


    def find_curve_max(self):
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['full']
            idx = np.argmax(_c[:, 3])
            self.map['d_curves'][d]['max_idx'] = idx


    def make_monokV_curve(self, kV):
        # suche in jeder Kurve den index der np.where(curve == kV) ist
        X = []
        Y = []
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['full']
            idx = np.where(_c[:, 0] == kV)[0]
            x_val = _c[:, 1][idx]
            y_val = _c[:, 3][idx]
            X.append(x_val[0])
            Y.append(y_val[0])
        return np.vstack((X, Y)).T


    def filter_relevant_curves(self, T_val):
        rel_curves = {}
        for d in self.map['d_curves']:

            _c_T = self.map['d_curves'][d]['full'][:, 1]
            _c_SNR = self.map['d_curves'][d]['full'][:, 3]

            if _c_T[0] <= T_val <= _c_T[-1]:
                rel_curves[d] = self.map['d_curves'][d]
        return rel_curves


    def create_monoKV_curve(self, kV_val):
        monokV_points = self.make_monokV_curve(kV=kV_val)
        curve_fit, data_points = self.vertical_interpolation(points=monokV_points)
        return curve_fit, data_points[::-1]


    def mono_kv_curve(self, U_val):
        T = []
        SNR = []

        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['full']
            kv = _c[:, 0]
            transmission = _c[:, 1]
            snr = _c[:, 3]

            val, idx = h.find_nearest(array=kv, value=U_val)
            T.append(transmission[idx])
            SNR.append(snr[idx])

        T = np.asarray(T)
        SNR = np.asarray(SNR)
        return T, SNR



    def find_intercept(self, kv_val, T_val):
        """
        find_intercept() searches for T(U0)-T_min intercept
        The function iterates through each curve and compares the difference (delta) between actual curve transmission
        value and the previous one AT given index (idx / voltage value). --> searches for min. deviation. The curve with
        minimal deviation corresponds to the searched thickness _d.

        returns T, snr, d

        :param c1:  expects an array as input.

        :param c2:  expects an array or a value. This allows more flexibility in searching for intercepts
                    between curves and curves or between curves and const. values.

        """
        old_delta = None
        _d = None
        idx = None

        rel_curves = self.filter_relevant_curves(T_val=T_val)

        for d in rel_curves:
            _c = rel_curves[d]['full']
            kv = _c[:, 0]
            T = _c[:, 1]

            _, idx = h.find_nearest(array=kv, value=kv_val)

            delta = np.abs(T[idx] - T_val).min()
            if old_delta is None:
                old_delta = delta
            elif delta < old_delta:
                old_delta = delta
                _d = d

        isnr = self.map['d_curves'][_d]['full'][:, 3][idx]
        iT = self.map['d_curves'][_d]['full'][:, 1][idx]

        return iT, isnr, _d


    def create_U0_curve(self, U0):
        self.map['U0_curve'] = {}
        T, SNR = self.mono_kv_curve(U_val=U0)
        self.map['U0_curve']['U0_val'] = U0
        self.map['U0_curve']['raw_data'] = self.Generator.merge_data(T, SNR)


    def create_Ubest_curve(self, d):
        _c = self.map['d_curves'][d]['full']
        kv = _c[:, 0]
        snr = _c[:, 3]
        max_val, idx = h.find_max(array=snr)
        kv_opt = kv[idx]

        self.map['Ubest_curve'] = {}
        T, SNR = self.mono_kv_curve(U_val=kv_opt)
        self.map['Ubest_curve']['Ubest_val'] = kv_opt
        self.map['Ubest_curve']['raw_data'] = self.Generator.merge_data(T, SNR)



    def create_lookup_table(self):

        iT, isnr, id, theta = self.translate_T_to_d()

        t_exp = self.calc_t_exp(snr_arr=isnr)

        plt.plot(theta, t_exp)
        plt.show()

        print('test')


    def smooth_curve(self, arr_1, arr_2):
        pass


    def translate_T_to_d(self):
        Ubest = self.map['Ubest_curve']['Ubest_val']
        transmission = self.fast_CT_data[0]
        angles = self.fast_CT_data[1]

        list_iT = []
        list_isnr = []
        list_id = []
        list_theta = []

        for i in range(len(angles)):
            # 1) find intercept between T(theta) and Ubest
            theta = angles[i]
            T = transmission[i]
            iT, isnr, id = self.find_intercept(kv_val=Ubest, T_val=T)
            list_iT.append(iT)
            list_theta.append(theta)
            list_isnr.append(isnr)
            list_id.append(id)

        return np.asarray(list_iT), np.asarray(list_isnr), np.asarray(list_id), np.asarray(list_theta)



    def calc_t_exp(self, snr_arr):
        t_exp = []
        for i in range(snr_arr.shape[0]):
            tmp_t = snr_arr[i] / self.snr_user
            t_exp.append(tmp_t)
        return np.asarray(t_exp)







    def printer(self):
        ix = self.map['iU0']['x']
        iy = self.map['iU0']['y']
        d_opt = self.map['iU0']['d']
        kV_opt = self.map['Ubest_curve']['Ubest_val']

        print(f'\n'
              f'intercept T_min o--/--o U0:\n' f'({round(ix, 3)} / {round(iy, 3)})\n'
              f'\n'
              f'interpolated thickness at intercept (d_opt):\n'  f'{d_opt}\n'
              f'\n'
              f'optimal voltage for measurement:\n' + f'{kV_opt} kV' + '\n')

