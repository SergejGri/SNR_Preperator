import csv
import gc
import time
import numpy as np
import os
import matplotlib.pyplot as plt
import helpers as h
from Plots import Plotter


class SNRMapGenerator:
    def __init__(self, scanner: object, d: list, kV_filter: list = None):
        """
        :param path_snr:
        :param path_T:
        :param path_fin:
        :param d:
        :param kV_filter:
        """
        self.scanner = scanner
        self.path_snr = self.scanner.p_SNR_files
        self.path_T = self.scanner.p_T_files
        self.path_fin = os.path.join(os.path.dirname(self.path_T), 'MAP')

        self.ds = self.scanner.files['ds']
        self.kVs = self.find_kvs()
        self.str_d = None
        self.T_min = None
        self.curves = {}
        self.ROI = {}
        self.MAP_object = {}
        self.U0_curve = None
        self.d_opt = None
        self.opt_curve = None

        if kV_filter is not None:
            self.kV_filter = kV_filter
            print(f'You passed {self.kV_filter} as a kV filter.')
        else:
            self.kV_filter = None
            print(f'No value for kV_filter was passed. All voltage folders are being included for evaluation.')


    def __call__(self, spatial_range, *args, **kwargs):
        self.check_range(spatial_range)
        self.MAP_object['ROIs'] = self.ROI


        for i in range(len(self.ds)):
            # 1)    look into base_path and find the used voltages to get the steps between kV_{i+1} and kV_{i}
            # 2)    since only whole num voltages are allowed, divide the range between kV_{i+1} and kV_{i} into one
            #       step ranges: Range from 40kV - 180kV -> 141 steps (including the very last step 140+1)
            # 3)    create function which calls the h.give_steps for every pair of kv and slice it to a curve
            int_d = self.ds[i]
            self.str_d = f'{self.ds[i]}_mm'
            self.curves[float(f'{int_d}')] = {}


            kV, T = self.get_T_data()
            SNR = self.get_SNR_data(int_d, self.ROI['lb'], self.ROI['rb'])


            self.curves[float(f'{int_d}')]['data'] = self.merge_data(kV=kV, T=T, SNR=SNR)

            kVs = self.create_kV_curve(int_d)
            self.curves[float(f'{self.ds[i]}')]['fit'] = self.merge_data(kV=kV_fit, T=x, SNR=y)

        self.MAP_object['d_curves'] = self.curves
        self.write_curve_files(self.curves)

        return self.MAP_object

    def get_T_data(self):
        data_T = np.genfromtxt(os.path.join(self.path_T, f'{self.str_d}.csv'), delimiter=';')
        data_T = data_T[data_T[:, 0].argsort()]
        data_T = np.asarray(data_T)

        if self.kV_filter is not None:
            for v in self.kV_filter:
                data_T = data_T[data_T[:, 0] != v]

        return data_T[:, 0].T, data_T[:, 1].T

    def get_SNR_data(self, d, lb, rb):
        kvs = []
        snr_means = []

        for file in self.scanner.files['SNR']:
            _d = h.extract_d(file)
            if _d == d:
                kV, mean_SNR = self.calc_avg_SNR(file, lb, rb)
                kvs.append(kV)
                snr_means.append(mean_SNR)

        kv_arr = np.asarray(kvs).T
        snr_arr = np.asarray(snr_means).T
        arr = np.vstack((kv_arr, snr_arr)).T
        arr = arr[arr[:, 0].argsort()]
        return arr[:, 1]

    def calc_avg_SNR(self, file, lb, rb):
        # read the file which is produced by the script SNR_Spectra.py
        # interpolate between data points, because for initial MAP there are to little data points between the first and
        # second entry. The data points are not equally distributed.
        kv = h.extract_kv(file)

        data = np.genfromtxt(file, skip_header=3)
        data = self.interpolate_data(data)

        data_u = data[:, 0]
        data_x = 1 / (2 * data_u)
        data = np.c_[data, data_x]
        data = data[np.logical_and(data[:, 4] >= lb, data[:, 4] <= rb)]
        mean_SNR = data[:, 1].mean()
        return kv, mean_SNR

    def merge_data(self, kV, T, SNR):
        d_curve = np.vstack((kV, T, SNR)).T
        d_curve.astype(float)
        return d_curve

    def interpolate_data(self, data, idx: tuple=None):
        """
        :param data:    have to be the same format as files which are produced by the SNR_spectra.py script.
                        -> Four columns. Here it is assumed that the columns are: (u, SNR, SPS, NPS).
        :param idx:     first index for starting the interpolation. If idx is None, full curve interpolation will be
                        performed.
        """
        u = data[:, 0]
        SNR = data[:, 1]
        SPS = data[:, 2]
        NPS = data[:, 3]

        if idx is not None:
            xvals = np.linspace(u[idx[0]], u[idx[-1]], 10)      # interpolate data between very first and second entry
            inter_SNR = np.interp(xvals, u, SNR)
            inter_SPS = np.interp(xvals, u, SPS)
            inter_NPS = np.interp(xvals, u, NPS)

            u = np.insert(u, 1, xvals).reshape(-1, 1)           # inserting the interpolated values into data and reshape them for easier merging
            SNR = np.insert(SNR, 1, inter_SNR).reshape(-1, 1)
            SPS = np.insert(SPS, 1, inter_SPS).reshape(-1, 1)
            NPS = np.insert(NPS, 1, inter_NPS).reshape(-1, 1)
            data = np.concatenate((u, SNR, SPS, NPS), axis=1)
            return data[1:, :]                                  # removing very first row (duplicate from interpolation + inserting)
        else:
            xvals = np.linspace(u[0], u[-1], 250).reshape(-1, 1)
            inter_SNR = np.interp(xvals, u, SNR).reshape(-1, 1)
            inter_SPS = np.interp(xvals, u, SPS).reshape(-1, 1)
            inter_NPS = np.interp(xvals, u, NPS).reshape(-1, 1)
            data = np.concatenate((xvals, inter_SNR, inter_SPS, inter_NPS), axis=1)
            return data

    def find_kvs(self):
        kvs = []
        for tfile in os.listdir(self.path_T):
            with open(os.path.join(self.path_T, tfile), mode='r') as f:
                for row in f:
                    kvs.append(row.split(';')[0])
            break
        kvs = sorted(kvs, key=lambda x: int(x))
        return kvs


    def piecewise_interpolatio(self):
        pass



    def reset(self):
        self.str_d = {}
        self.curves = {}


    def write_curve_files(self, curves):
        for c in curves:
            c = int(c)
            if not os.path.isdir(self.path_fin):
                os.makedirs(self.path_fin)
            np.savetxt(os.path.join(self.path_fin, f'{c}mm_fit.csv'), self.curves[c]['fit'], delimiter=',')
            np.savetxt(os.path.join(self.path_fin, f'{c}mm_data_points.csv'), self.curves[c]['data'], delimiter=',')


    def pick_value(self):
        pass


    def units_converter(self, val):
        """
        :param:
        """
        val = val * 10 ** (-6)
        return val


    def check_range(self, rng):
        if isinstance(rng, int):
            rng = (rng,)[0]
            lb = rng - 0.1 * rng
            rb = rng + 0.1 * rng
        else:
            lb = rng[0]
            rb = rng[1]
        self.ROI['lb'] = lb
        self.ROI['rb'] = rb

    def poly_fit(self, var_x, var_y, steps):
        a, b, c = np.polyfit(var_x, var_y, deg=2)
        x = np.linspace(var_x[0], var_x[-1], steps)
        y = self.func_poly(x, a, b, c)
        return x, y

    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c

    @staticmethod
    def get_properties(file):
        filename = os.path.basename(file)
        try:
            kv = h.extract_kv(filename)
        except ValueError:
            print('check naming convention of your passed files.')
        return kv

    def create_kV_curve(self, d):
        for i in range(len(self.kVs)-1):
            _p0 = []
            _p1 = []
            kv0 = int(self.kVs[i])
            kv1 = int(self.kVs[i+1])
            n = abs(int(kv1) - int(kv0))
            # now you need to get the T values with the index of _p0 and _p1
            _curve = self.curves[float(f'{self.ds[i]}')]['data']
            row0 = _curve[_curve[:, 0] == kv0]
            row1 = _curve[_curve[:, 0] == kv1]

            _p0 = [row0[:, 1][0], row0[:, 2][0]]
            _p1 = [row1[:, 1][0], row1[:, 2][0]]

            data_points = h.give_steps(p0=_p0, p1=_p1, pillows=n)

            plt.plot(_curve[:, 1], _curve[:, 2])
            plt.scatter(data_points[:, 0], data_points[:, 1])
            plt.show()
            print('test')


# TODO: implement a robust curve- / thickness-chose-mechanism
def plot(path_map, excl_filter=None):
    if not os.path.exists(os.path.join(path_map, '../Plots')):
        os.mkdir(os.path.join(path_map, '../Plots'))
    for file in os.listdir(path_map):
        if file.endswith('.csv') and not file.split('.')[0] in excl_filter:
            filename = file.split('m')[0]
            data = np.genfromtxt(os.path.join(path_map, file), delimiter=',')
            max_kv = data[-1][0]
            data_x = data[:, 1]
            data_y = data[:, 2]
            plt.figure(figsize=(14.4, 8.8))
            plt.plot(data_x, data_y, marker='o', label=f'{filename} mm')
            plt.legend()
            plt.xlabel('Transmission a.u.')
            plt.ylabel('SNR')
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(path_map, '../Plots'), f'SNR_T_{filename}mm_{max_kv}maxkV.png'))


def write_data(path_T, path_SNR, path_fin):
    now = time.strftime('%c')
    if not os.path.exists(os.path.join(path_fin, '../Plots')):
        os.makedirs(os.path.join(path_fin, '../Plots'))
    with open(os.path.join(path_fin, '../Plots', 'evaluation.txt'), 'w+') as f:
        f.write(f'{now}\n')
        f.write(f'used transmission data: {path_T}\n')
        f.write(f'used SNR data: {path_SNR}\n')
        f.close()
