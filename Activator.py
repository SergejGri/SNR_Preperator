import os
import gc
import csv
import sys

import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

from snr_calc.ct_operations import CT
from snr_calc.ct_operations import avg_multi_img_CT
from visual.Plotter import Plotter as PLT
from visual.Plotter import TerminalColor as PCOL
from snr_calc.map_generator import SNRMapGenerator
import helpers as hlp


class Scanner:
    def __init__(self, params):
        self.p_SNR_files = params.paths['snr_data']
        self.p_T_files = params.paths['T_data']
        self.ds_ex = params.excluded_thicknesses
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
    def __init__(self, attributes):
    #def __init__(self, paths: dict, U0: int, snr_user: float, base_texp: int = None, kv_ex: list = None,
    #             ds_ex: list = None, spatial_size=None, vir_curve_step: float = None):
        self.paths = attributes.paths
        self.fCT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None}
        self.fCT_texp = None
        self.CT_data = None
        self.CT_texp = None
        self.CT_theta = None
        self.T_min = None
        self.mode_avg = False
        if attributes.base_texp is not None:
            self.mode_avg = True
            self.base_texp = attributes.base_texp


        self.kv_ex = attributes.excluded_kvs
        self.ds_ex = attributes.excluded_thicknesses

        if attributes.spatial_size:
            self.ssize = attributes.spatial_size
        else:
            self.ssize = (100)
            self.init_MAP = True
            print('No spatial_size value was passed: Initial MAP creation @ 100E-6 m')
        self.snr_user = attributes.snr_user
        self.scanner = Scanner(params=attributes)

        self.stop_exe = False

        if 40 <= attributes.U0 <= 180:
            self.U0 = attributes.U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {attributes.U0} \n'
                  + PCOL.BOLD + '...exit...' + PCOL.END)
            sys.exit()

        self.kV_interpolation = False

        if attributes.virtual_curve_step is None:
            self.v_curve_step = 0.1
        else:
            self.v_curve_step = attributes.virtual_curve_step

        self.U0_intercept = {'x': {}, 'y': {}, 'd': {}}

        self.Ubest_curve = {'val': None, 'fit': {}, 'data': {}}
        self.U0_curve = {'val': self.U0, 'fit': {}, 'raw_data': {}}

        self.Generator = SNRMapGenerator(scanner=self.scanner, kv_filter=self.kv_ex)
        self._plt = PLT()


    def __call__(self, create_plot: bool = True, detailed: bool = False, just_fCT: bool = False):
        # 0) find U_best
        # 1) fast_CT
        # 2) extract T_min
        # 3)

        self.evaluate_fCT()


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
            self._plt.create_T_kv_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)
            self._plt.create_MAP_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)


        self.get_texp_from_fCT(mode_avg=self.mode_avg)
        self.evaluate_CT()
        self.CT_data = CT(path_ct=self.paths['CT_imgs'],
                          path_refs=self.paths['CT_refs'],
                          path_darks=self.paths['CT_darks'])


        self.t_exp, self.theta = self.create_lookup_table()



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


    def get_texp_from_fCT(self, mode_avg: bool):
        Ubest = self.map['Ubest_curve']['Ubest_val']
        fCT_T = self.fCT_data['T']
        fCT_theta = self.fCT_data['theta']

        self.fCT_data['T'], self.fCT_data['snr'], self.fCT_data['d'],  self.fCT_data['theta'] = \
            self.extract_MAP_data(kv_val=Ubest, transmission=fCT_T, angles=fCT_theta)
        
        self.fCT_data['texp'] = self.calc_texp(snr_arr=self.fCT_data['snr'])

        if mode_avg:
            self.fCT_data['avg_num'] = dict()
            self.fCT_data['avg_num'] = self.calc_avg(self.base_texp)


    def calc_avg(self, btexp):
        fCT_texp = self.fCT_data['texp']
        fCT_theta = self.fCT_data['theta']

        loc_avgs = []
        for i in range(fCT_texp.size):
            avg_nmum = hlp.round_to_nearest_hundred(btexp, fCT_texp[i])
            loc_avgs.append(avg_nmum)

        return np.asarray(loc_avgs)


    def extract_MAP_data(self, kv_val, angles: np.ndarray, transmission: np.ndarray):
        list_iT, list_isnr, list_id, list_theta = [], [], [], []

        for i in range(len(angles)):
            # 1) find intercept between T(theta) and Ubest
            theta = round(angles[i], 2)
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
            tmp_t = self.snr_user / snr_arr[i]
            t_exp.append(tmp_t)
        return np.asarray(t_exp)


    def evaluate_fCT(self):
        self.fCT_data['T'], self.fCT_data['theta'] = CT(path_ct=self.paths['fCT_imgs'],
                                                        path_refs=self.paths['fCT_refs'],
                                                        path_darks=self.paths['fCT_darks'])

        self.T_min, _ = hlp.find_min(self.fCT_data['T'])


        #hier muss noch eine Funtion die das gleiche wie bei evaluate_CT macht also texp calcen aber erst nachdem MAP usw gebaut wurde!


    def evaluate_CT(self):
        if self.mode_avg:

            self.CT_data['avg_num'] = self.interpolate_avg_num()

            splitted_cts = avg_multi_img_CT(self, self.paths['CT_imgs'], imgs_per_angle=4)


        # die interpolation muss vor der CT auswertung gemacht werden. Die Auswertung muss schon 'wissen' wie viel
        # sie pro winkelschritt avaregen soll.

        self.CT_data['T'], self.CT_data['theta'] = CT(path_ct=self.paths['CT_imgs'],
                                                        path_refs=self.paths['CT_refs'],
                                                        path_darks=self.paths['CT_darks'])

        Ubest = self.map['Ubest_curve']['Ubest_val']




        self.CT_data['T'], \
        self.CT_data['snr'], \
        self.CT_data['d'], \
        self.CT_data['theta'] = self.extract_MAP_data(kv_val=Ubest,
                                                       transmission=self.CT_data['T'],
                                                       angles=self.CT_data['theta'])

        self.CT_data['texp'] = self.calc_texp(snr_arr=self.CT_data['snr'])


    def interpolate_avg_num(self):
        fCT_avg = self.fCT_data['avg_num']
        fCT_theta = self.fCT_data['theta']

        step = 360 / 1500
        CT_theta = np.arange(0, 360, step)
        CT_avg = hlp.nearest_interp(xi=CT_theta, x=fCT_theta, y=fCT_avg)

        return CT_avg





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
