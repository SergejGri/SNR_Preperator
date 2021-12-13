import os
import gc
import csv
import sys

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from snr_calc.ct_evaluation import CT
from visual.Plotter import Plotter as PLT
from visual.Plotter import TerminalColor as PCOL
from snr_calc.map_generator import SNRMapGenerator
import helpers as hlp


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
                    d = hlp.extract(what='d', dfile=file)
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
                d = hlp.extract(what='d', dfile=file)
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

        self.fCT_data = None
        self.CT_data = None
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
        # 0) find U_best
        # 1) fast_CT
        # 2) extract T_min
        # 3)
        self.fCT_data = CT(path_ct=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\fast_CT\ct',
                           path_refs=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\fast_ct\refs',
                           path_darks=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\fast_ct\darks',
                           num_proj=1500)

        self.T_min, _ = hlp.find_min(self.fCT_data[:, 1])
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

        self.CT_data = CT(path_ct=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\ct',
                          path_refs=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\refs',
                          path_darks=r'\\132.187.193.8\junk\sgrischagin\2021-11-30-sergej-CT-halbesPhantom-5W-M5p3\darks',
                          num_proj=1500)



        self.create_lookup_table()


    def vertical_interpolation(self, points: np.array):
        step = 100000
        _f = interpolate.interp1d(points[:, 0], points[:, 1], kind='linear')
        x_fit = np.linspace(points[:, 0][0], points[:, 0][-1], step)
        y_fit = _f(x_fit)
        data = np.vstack((_f.x, _f.y)).T
        curve = np.vstack((x_fit, y_fit)).T
        return curve, data

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

            val, idx = hlp.find_nearest(array=kv, value=U_val)
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

            _, idx = hlp.find_nearest(array=kv, value=kv_val)

            delta = np.abs(T[idx] - T_val).min()
            if old_delta is None:
                old_delta = delta
            elif delta < old_delta:
                old_delta = delta
                _d = d

        isnr = self.map['d_curves'][_d]['full'][:, 3][idx]
        iT = self.map['d_curves'][_d]['full'][:, 1][idx]

        return iT, isnr, _d



    def create_lookup_table(self):

        Ubest = self.map['Ubest_curve']['Ubest_val']
        fCT_T, fCT_snr, fCT_d, fCT_theta = self.extract_MAP_data(kv_val=Ubest,
                                                                 transmission=self.fCT_data[:, 1],
                                                                 angles=self.fCT_data[:, 0])
        fCT_texp = self.calc_texp(snr_arr=fCT_snr)


        CT_T, CT_snr, CT_d, CT_theta = self.extract_MAP_data(kv_val=Ubest,
                                                             transmission=self.CT_data[:, 1],
                                                             angles=self.CT_data[:, 0])
        CT_texp = self.calc_texp(snr_arr=CT_snr)





        fig, (ax1, ax2) = plt.subplots(2)

        ax1.plot(fCT_theta, fCT_texp, label=r'$t_{exp}(\theta)$ (fCT)')
        ax1.scatter(fCT_theta, fCT_texp)
        ax1.plot(CT_theta, CT_texp, label=r'$t_{exp}(\theta)$ (CT)')
        ax1.scatter(CT_theta, CT_texp)
        ax1.set_ylabel('$t_{exp}$  [ms]')
        ax1.set_xlabel(r'$\theta$  $[\circ]$')
        ax1.legend()

        ax2.set_title('ax2 title')
        ax2.plot(fCT_theta, fCT_d, label=r'$d(\theta)$ (fCT)')
        ax2.scatter(fCT_theta, fCT_d)
        ax2.plot(CT_theta, CT_d, label=r'$d(\theta)$ (CT)')
        ax2.scatter(CT_theta, CT_d)
        ax2.set_ylabel('sample thickness [mm]')
        ax2.set_xlabel(r'$\theta$  $[\circ]$')

        ax2.legend()
        fig.tight_layout()
        plt.show()

        fig.savefig(os.path.join(self.scanner.path_fin, 'plots', f'd_t_theta-usr_snr_{self.snr_user}.pdf'), dpi=600)
        print('test')

    def smooth_curve(self, arr_1, arr_2):
        pass

    def create_U0_curve(self, U0):
        self.map['U0_curve'] = {}
        T, SNR = self.mono_kv_curve(U_val=U0)
        self.map['U0_curve']['U0_val'] = U0
        self.map['U0_curve']['raw_data'] = self.Generator.merge_data(T, SNR)

    def create_Ubest_curve(self, d):
        _c = self.map['d_curves'][d]['full']
        kv = _c[:, 0]
        snr = _c[:, 3]
        max_val, idx = hlp.find_max(array=snr)
        kv_opt = kv[idx]

        self.map['Ubest_curve'] = {}
        T, SNR = self.mono_kv_curve(U_val=kv_opt)
        self.map['Ubest_curve']['Ubest_val'] = kv_opt
        self.map['Ubest_curve']['raw_data'] = self.Generator.merge_data(T, SNR)

    def extract_MAP_data(self, kv_val, angles: np.ndarray, transmission: np.ndarray):
        list_iT, list_isnr, list_id, list_theta = [], [], [], []

        #Ubest = self.map['Ubest_curve']['Ubest_val']
        #transmission = self.fast_CT_data[:, 1]
        #angles = self.fast_CT_data[:, 0]

        for i in range(len(angles)):
            # 1) find intercept between T(theta) and Ubest
            theta = angles[i]
            T = transmission[i]
            iT, isnr, id = self.find_intercept(kv_val=kv_val, T_val=T)
            list_iT.append(iT), list_theta.append(theta), list_isnr.append(isnr), list_id.append(id)

        return np.asarray(list_iT), np.asarray(list_isnr), np.asarray(list_id), np.asarray(list_theta)

    def calc_texp(self, snr_arr):
        """
        Since the SNR values are SNR/s (id did not
        """
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
