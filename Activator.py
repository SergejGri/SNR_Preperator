import os
import gc
import csv
import collections
from matplotlib import pyplot as plt

from Plots.Plotter import Plotter as PLT
from Plots.Plotter import TerminalColor as PCOL
from SNR_Calculation.map_generator import SNRMapGenerator
from scipy import interpolate
import numpy as np


class Scanner:
    def __init__(self, snr_files: str, T_files: str):
        self.p_SNR_files = snr_files
        self.p_T_files = T_files
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
                    d = self.extract_d(file)
                    if f'_{d}_mm_' in file:
                        loc.append(os.path.join(_subdir, file))
        self.files['SNR'] = loc

    def collect_transmission_files(self):
        loc = []
        for file in os.listdir(self.p_T_files):
            if file.endswith('.csv'):
                loc.append(os.path.join(self.p_T_files, file))
        self.files['T'] = loc

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

    @staticmethod
    def extract_d(file):
        d_str = file.split('kV_')[1]
        d_str = d_str.split('_mm')[0]
        return int(d_str)


class Curve:
    def __init__(self, d, kV: np.array, T: np.array, SNR: np.array):
        self.d = d
        self.kV = kV
        self.T = T
        self.SNR = SNR

        self.curve = {f'{self.d}': {}}
        self.curve[f'{self.d}']['kV'] = kV
        self.curve[f'{self.d}']['T'] = T
        self.curve[f'{self.d}']['SNR'] = SNR


class Activator():
    def __init__(self, data_T: np.array, snr_files: str, T_files: str, U0: int, ds: list, snr_user: float, ssize=None,
                 create_plot: bool = False):
        self.fast_CT_data = data_T
        self.T_min = self.get_min_T()
        if ssize:
            self.ssize = ssize
        else:
            self.ssize = (250, 150)
            self.init_MAP = True
        self.snr_user = snr_user
        self.scanner = Scanner(snr_files=snr_files, T_files=T_files)
        self.curves = []
        self.stop_exe = False

        if 40 <= U0 and U0 <= 180:
            self.U0 = U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {U0} \n'
                  + PCOL.BOLD + '...exit...' + PCOL.END)
            self.stop_exe = True

        self.kV_interpolation = False
        self.ds = ds
        self.x_U0_c = []
        self.y_U0_c = []
        self.x_U0_points = []
        self.y_U0_points = []

        self.intercept = {'x': {}, 'y': {}}
        self.intercept_found = False
        self.d_opt = None
        self.X_opt = None
        self.Y_opt = None
        #self.kV_opt = None
        self.opt_curve = {'val': None,'fit': {}, 'data': {}}
        self.opt_data_points = None
        self.map = None
        self.U0_curve = {'val': self.U0, 'fit': {}, 'data': {}}
        self.Generator = SNRMapGenerator(scnr=self.scanner, d=self.ds)

    def __call__(self, create_plot: bool = True, *args, **kwargs):

        self.map = self.Generator(spatial_range=self.ssize)
        self.map['T_min'] = self.T_min

        self.create_virtual_curves()

        self.U0_curve['fit'], self.U0_curve['data'] = self.create_monoKV_curve(kV_val=self.U0)

        self.map['U0_curve'] = self.U0_curve

        self.curve_minT_intercept(curve=self.U0_curve['fit'])

        if self.intercept_found:
            self.get_opt_SNR_curve()
            self.map['opt_curve'] = self.opt_curve

        self.calc_t_exp()
        self.printer()
        if create_plot:
            _plt = PLT()
            # _plt.create_plot(path_result=self.scanner.path_fin, object=map)
            _plt.create_v_plot(path_result=self.scanner.path_fin, object=self.map, full=True)

    def get_min_T(self):
        return np.min(self.fast_CT_data[0])

    # TODO: the very first approach of the code contained just lists as container for curves and datapoints. I started
    #  to rewrite all the code from the beginning to make dicts as the one container. Couldn't finish it because of to
    #  little time. That's why from here on there are lists and dicts mixed.
    #def read_curves(self, map):
    #    for d in map['d_curves']:
    #        kV = np.asarray(map['d_curves'][d][:, 0])
    #        T = np.asarray(map['d_curves'][d][:, 1])
    #        SNR = np.asarray(map['d_curves'][d][:, 2])
    #        self.curves.append(Curve(d=d, kV=kV, T=T, SNR=SNR))

    def interpolate_curve_piece(self, kV_val):
        # 1)    get left and right neighbours and interpolate between these values
        # 2)    create linspaced values and fit in the area between left and right neighbours
        # 3)    pick fitted x and fitted y value at given 'distance'
        # 4)    distance: e.g. U0 is set to 67kV then it is 7 steps away from left neighbour 60. That's way it it is
        #       needed to pick the 7th entry in the fitted range to get the desired kV value for every curve. Than
        #       a 'vertical' curve can be 'created'.

        dist, lb, rb, step = self.find_neighbours(kV_val)
        x_points = []
        y_points = []


        for d in self.map['d_curves']:
            curve = self.map['d_curves'][d]
            _c_kV = curve[:, 0]
            _c_T = curve[:, 1]
            _c_SNR = curve[:, 2]

            il = np.where(_c_kV == lb)[0][0]
            ir = np.where(_c_kV == rb)[0][0]

            a, b, c = np.polyfit(_c_T, _c_SNR, deg=2)
            x_curve_fit = np.linspace(_c_T[il], _c_T[ir], step + 1)
            y_curve_fit = self.func_poly(x_curve_fit, a, b, c)

            x_points.append(x_curve_fit[dist])
            y_points.append(y_curve_fit[dist])

        return np.vstack((x_points, y_points)).T

    def vertical_interpolation(self, points: np.array):
        step = 10000
        _f = interpolate.interp1d(points[:, 0], points[:, 1], kind='linear')
        x_fit = np.linspace(points[:, 0][0], points[:, 0][-1], step)
        y_fit = _f(x_fit)
        data = np.vstack((_f.x, _f.y)).T
        curve = np.vstack((x_fit, y_fit)).T
        return curve, data

    def curve_minT_intercept(self, curve):
        # TODO: need a finally statement at the try/except block -> worst case is stopt the execution or pass 'standard values'?
        # TODO: more robust idx calculation. Catching cases like 1 < len(idx). ->

        epsilons = [0.000001, 0.00001, 0.0001]
        for eps in epsilons:
            idx = np.where((curve[:, 0] > (self.T_min - eps)) & (curve[:, 0] < (self.T_min + eps)))[0]
            if idx.size:
                self.intercept['x'] = self.T_min
                self.intercept['y'] = curve[:, 1][idx[0]]
                self.intercept_found = True
                break

    def create_virtual_curves(self):
        #   1) read first and second curve
        #   2) calc the number of curves which needed to be created between first and second in respect to the step size
        #   3) take the first data point (SNR/kV) of the second curve and the first data point (SNR/kV) of the first
        #      curve and divide the abs between them into c_num + 1 pieces
        #   4) go trough every data point in T_1 and T_2 (or SNR since len(T)=len(SNR)), interpolate 'linear' between
        #   5) linspace the 'distance' between the data points linspace(T_2 - T_1)
        #   6) doing this for every pair of curves and its data points, you will get a 'grid' of virtual data points
        step = 0.1
        temp_curves = {}
        for d, i in zip(self.map['d_curves'], range(len(self.ds)-1)):
            X = []
            Y = []
            c_num, sorted_ds = self.calc_curve_num(d, step)

            d2 = sorted_ds[i + 1]
            d1 = sorted_ds[i]
            _c2 = self.map['d_curves'][d2]
            _c1 = self.map['d_curves'][d1]
            kV_2, T_2, SNR_2 = _c2[:, 0], _c2[:, 1], _c2[:, 2]
            kV_1, T_1, SNR_1 = _c1[:, 0], _c1[:, 1], _c1[:, 2]

            for j in range(len(T_1)):
                _x = [T_2[j], T_1[j]]
                _y = [SNR_2[j], SNR_1[j]]
                f = interpolate.interp1d(_x, _y, kind='linear')
                _x_new = np.linspace(T_2[j], T_1[j], len(c_num) + 2)[1:-1]
                _y_new = f(_x_new)
                X.append(_x_new)
                Y.append(_y_new)

            for k in range(c_num.size):
                _T = []
                _SNR = []
                kV = kV_1
                _d = round(c_num[k], 2)
                for _j in range(len(T_1)):
                    _T.append(X[_j][k])
                    _SNR.append(Y[_j][k])

                kV = np.asarray(kV)
                _T = np.asarray(_T)
                _SNR = np.asarray(_SNR)
                merged_curve = self.Generator.merge_data(kV=kV, T=_T, SNR=_SNR)
                temp_curves[_d] = merged_curve

        self.map['d_curves'].update(temp_curves)
        self.map['d_curves'] = dict(sorted(self.map['d_curves'].items()))


    def find_neighbours(self, kV):
        # find first element in map where arg. >= self.U0 ==> right border
        # the nearest left entry is the left border between which the interpolation will take place
        key = None
        for key in self.map['d_curves']:
            break
        _c_kV = self.map['d_curves'][key][:, 0].tolist()

        num = next(i[0] for i in enumerate(_c_kV) if i[1] >= kV)
        left_nbr = _c_kV[num-1]
        right_nbr = _c_kV[num]
        dist = kV - left_nbr

        return abs(int(dist)), int(left_nbr), int(right_nbr), (int(right_nbr) - int(left_nbr))

    def get_opt_SNR_curve(self):
        #   1) filter all relevant curves
        #   2) find min abs between x value of curves and intercept
        #   3) look into every watch_curve. If max SNR value is smaller than the actual intercept SNR value - > next curve
        #      to find _d_max
        #   4) 141 step size is because of the voltage difference 180kV-40kV. This
        relevant_curves = self.filter_relevant_curves()

        old_delta = None
        for d in relevant_curves:
            _c = relevant_curves[d]
            _x, _y = self.poly_fit(_c[:, 1], _c[:, 2], 10000)

            #   1) estimate the nearest interpolated x values to the intercept_x
            idx = (np.abs(_x - self.T_min)).argmin()

            delta = abs(_y[idx] - self.intercept['y'])
            if old_delta is None:
                old_delta = delta
            if delta < old_delta:
                old_delta = delta
                self.map['d_opt'] = d

        self.find_opt_curve()


    def filter_relevant_curves(self):
        watch_curves = {}
        for d in self.map['d_curves']:
            _c_SNR = self.map['d_curves'][d][:, 2]
            _SNR_max = max(_c_SNR)
            if not max(_c_SNR) < self.intercept['y']:
                watch_curves[d] = self.map['d_curves'][d]
        return watch_curves

    def find_opt_curve(self):
        _c = self.map['d_curves'][self.map['d_opt']]
        _c_kV = _c[:, 0]
        _c_T = _c[:, 1]
        _c_SNR = _c[:, 2]
        steps = int(_c_kV[-1] - _c_kV[0]) + 1
        kVs = np.linspace(_c_kV[0], _c_kV[-1], steps)
        x, y = self.poly_fit(_c_T, _c_SNR, steps)
        idx = np.argmax(y)
        self.opt_curve['val'] = kVs[idx]

        self.opt_curve['fit'], self.opt_curve['data'] = self.create_monoKV_curve(kV_val=kVs[idx])



    def search_nearest_curve(self):
        # 1) accept transmission array from fast_ct() t_arr = [[p1, p2,..], [T1, T2 ..]]
        # 2) translate from T(proj) to d. There is a need in continuous d values
        pass

    def create_monoKV_curve(self, kV_val):
        curve_points = self.interpolate_curve_piece(kV_val=kV_val)
        curve_fit, data_points = self.vertical_interpolation(points=curve_points)
        return curve_fit, data_points[::-1]

    def calc_t_exp(self):
        t_exp = 1
        return t_exp

    def calc_look_up_table(self):
        #   1) create MAP in ROI (the map_object should carry real data points, and the fitted curves
        #   2) find intercept between T(theta) and the U_best curve. Intercept corresponds to d(theta)
        #   3)
        look_up = {'T': self.fast_CT_data[0], 'theta': self.fast_CT_data[1], 'd': {}, 'SNR': {}, 't_exp': {}}
        pass

    def printer(self):
        icpt_x = self.intercept['x']
        icpt_y = self.intercept['y']
        d_opt = self.map['d_opt']
        kV_opt = self.map['opt_curve']['val']

        if self.intercept_found == True:
            print(f'intercept T_min and U0:\n'
                  f'({round(icpt_x, 3)} / {round(icpt_y, 3)})')
            print(' ')
            print(f'interpolated thickness at intercept (d_opt):\n'
                  f'{d_opt}')
            print(' ')
            print('optimal voltage for measurement:\n'
                   + f' ==> kV_opt = {kV_opt} kV <==' + '\n')
        else:
            print('No intercept between U0 and T_min could be found. \n'
                  '-> You may reduce epsilon in find_intercept()')

    def poly_fit(self, var_x, var_y, steps):
        a, b, c = np.polyfit(var_x, var_y, deg=2)
        x = np.linspace(var_x[0], var_x[-1], steps)
        y = self.func_poly(x, a, b, c)
        return x, y

    def create_supervision_plot(self):
        for curve in self.curves:
            a, b, c = np.polyfit(curve.T, curve.SNR, deg=2)
            x = np.linspace(curve.T[0], curve.T[-1], 141)
            y = self.func_poly(x, a, b, c)
            plt.plot(x, y, alpha=0.8, label=f'{curve.d}')
            plt.scatter(curve.T, curve.SNR)
        plt.plot(self.U0_curve[:, 0], self.U0_curve[:, 1])
        plt.axvline(x=self.T_min, c='green', linestyle='--', alpha=0.5, linewidth=1)
        # plt.yscale('log')
        plt.legend()
        plt.savefig(os.path.join(f'{self.scanner.path_fin}', 'plots', 'supervisor.pdf'))
        plt.show()

    def calc_curve_num(self, idx, step):
        idx = int(idx)
        sorted_ds = dict(sorted(self.map['d_curves'].items()))
        ds = [*sorted_ds]
        i = ds.index(idx)
        c_num = np.arange(ds[i], ds[i + 1], step)[1:]
        return c_num[::-1], ds

    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c
