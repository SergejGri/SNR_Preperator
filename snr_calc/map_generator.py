import time
import numpy as np
import os
import helpers as h
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P


class SNRMapGenerator:
    def __init__(self, scanner, d: list, kv_filter: list = None):
        """
        :param path_snr:
        :param path_T:
        :param path_fin:
        :param d:
        :param kv_filter:
        """
        self.scanner = scanner
        self.path_snr = self.scanner.p_SNR_files
        self.path_T = self.scanner.p_T_files
        self.path_fin = os.path.join(os.path.dirname(self.path_T), 'MAP')

        self.ds = self.scanner.files['ds']
        self.kVs = self.used_voltages(filter=kv_filter)
        self.str_d = None
        self.T_min = None
        self.curves = {}
        self.ROI = {}
        self.MAP_object = {'ds': self.ds}
        self.U0_curve = None
        self.d_opt = None
        self.opt_curve = None

        if kv_filter is not None:
            self.kV_filter = kv_filter
            print(f'You passed {self.kV_filter} as excluding voltage.')
        else:
            self.kV_filter = None
            print(f'No value for kV_filter was passed. All voltage folders are being included for evaluation.')


    def __call__(self, spatial_range):
        self.check_range(spatial_range)
        self.MAP_object['ROIs'] = self.ROI

        # 1) read T and SNR data from measurement
        # 2) create "raw curves" from 1) data
        for i in range(len(self.ds)):
            int_d = self.ds[i]
            self.str_d = f'{self.ds[i]}_mm'
            self.curves[float(f'{int_d}')] = {}

            kV, T = self.get_T_data()
            SNR = self.get_SNR_data(int_d, self.ROI['lb'], self.ROI['rb'])
            self.curves[float(f'{int_d}')]['raw_data'] = self.merge_data(kV, T, SNR)

        # 1) go through all curves and
        # 2) create virtual data points for SNR(T) curves between measured thicknesses
        self.create_raw_grid()
        for d in self.curves:
            full_curve = self.create_kv_grid(d)
            self.curves[float(d)]['full'] = full_curve

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
            _d = h.extract(what='d', dfile=file)
            if _d == d:
                kV, mean_SNR = self.calc_avg_SNR(file, lb, rb)
                if kV in self.kV_filter:
                    del kV, mean_SNR
                else:
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
        kv = h.extract(what='kv', dfile=file)
        data = np.genfromtxt(file, skip_header=3)
        data = self.interpolate_data(data)

        data_u = data[:, 0]
        data_x = 1 / (2 * data_u)
        data = np.c_[data, data_x]
        data = data[np.logical_and(data[:, 4] >= lb, data[:, 4] <= rb)]
        mean_SNR = data[:, 1].mean()
        return kv, mean_SNR

    def merge_data(self, *cols):
        d_curve = np.vstack((cols)).T
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


    def used_voltages(self, filter=None):
        kvs = []
        for tfile in os.listdir(self.path_T):
            with open(os.path.join(self.path_T, tfile), mode='r') as f:
                for row in f:
                    kvs.append(row.split(';')[0])
            break
        kvs = sorted(kvs, key=lambda x: int(x))
        if filter is not None:
            for fval in filter:
                if fval in filter:
                    kvs.remove(str(fval))
        return kvs



    def create_kv_grid(self, d):
        _c = self.curves[float(f'{d}')]['raw_data']
        kV = _c[:, 0]
        T = _c[:, 1]
        SNR = _c[:, 2]

        grid = np.empty(shape=3)

        for i in range(len(self.kVs) - 1):

            # TODO: change the selection of kvs depending on the curve
            kv0 = int(self.kVs[i])
            kv1 = int(self.kVs[i + 1])
            n = abs(int(kv1) - int(kv0) + 1)

            row0 = _c[kV == kv0]
            row1 = _c[kV == kv1]

            p0 = [row0[:, 1][0], row0[:, 2][0]]    # extract [T0, SNR0] and [T1, SNR1] for triangle
            p1 = [row1[:, 1][0], row1[:, 2][0]]

            range_kv_points = h.give_lin_steps(p0=p0, p1=p1, pillars=n)

            kvs_vals = np.linspace(kv0, kv1, n)[np.newaxis].T
            kvs_vals = np.hstack((kvs_vals, range_kv_points))
            grid = np.vstack((grid, kvs_vals))

        # fine tune new curve
        grid = grid[1:]
        del_rows = []
        for j in range(len(grid[:, 0]) - 1):
            if grid[j, 0] == grid[j + 1, 0]:
                del_rows.append(j)
        del_rows = np.asarray(del_rows)
        grid = np.delete(grid, [del_rows], axis=0)

        new_x_axis = grid[:, 1]
        semi_fit_SNR = grid[:, 2]
        fit_params = np.polyfit(new_x_axis, semi_fit_SNR, 2)
        self.curves[float(f'{d}')]['fit_params'] = fit_params
        f = np.poly1d(fit_params)
        fit_new_axis = f(new_x_axis)

        kv_steps_full = np.arange(kV[0], kV[-1]+1, step=1)
        kv_grid = self.merge_data(kv_steps_full, new_x_axis, semi_fit_SNR, fit_new_axis)

        return kv_grid




    def create_raw_grid(self):
        d_gap = 0.1
        temp_curves = {}

        for i in range(len(self.ds)-1):
            d1 = self.ds[i]
            d2 = self.ds[i+1]
            sub_ds = np.arange(d1, d2, d_gap)[1:]   # calc. num of curves between d1 and d2

            c1 = self.curves[d1]['raw_data']
            c2 = self.curves[d2]['raw_data']

            raw_virtual_points = self.data_points_interpolation(sub_ds=sub_ds, curve1=c1, curve2=c2)

            for d in raw_virtual_points:
                T = raw_virtual_points[d]['X']
                snr = raw_virtual_points[d]['Y']

                self.curves[d] = {}
                self.curves[d]['raw_data'] = self.merge_data(c1[:, 0], T, snr)
                #temp_curves[d] = {}
                #temp_curves[d]['raw_data'] = self.merge_data(c1[:, 0], T, snr)

        self.curves = dict(sorted(self.curves.items()))
        #self.MAP_object['d_curves'].update(temp_curves)
        #self.MAP_object['d_curves'] = dict(sorted(self.MAP_object['d_curves'].items()))




    def update_map(self):
        self.MAP_object = self.MAP_object








    def data_points_interpolation(self, sub_ds: list, curve1: np.asarray, curve2: np.array):
        """
        takes a list of thicknesses, two curves and calculates equidistant points between measured data points. I call
        these interpolated data points 'virtual' points
        """

        raw_virtual_points = {}
        kv1, T1, snr1 = curve1[:, 0], curve1[:, 1], curve1[:, 2]
        kv2, T2, snr2 = curve2[:, 0], curve2[:, 1], curve2[:, 2]

        X = []
        Y = []

        # pick T1, T2 and SNR1, SNR2 for triangle
        for j in range(len(T1)):
            x = [T1[j], T2[j]]
            y = [snr1[j], snr2[j]]

            coeff_t, coeff_m = P.polyfit(x, y, 1)
            xvals = np.linspace(T1[j], T2[j], len(sub_ds))
            yvals = h.linear_f(x=xvals, m=coeff_m, t=coeff_t)

            X.append(xvals)
            Y.append(yvals)

        X = np.asarray(X)
        Y = np.asarray(Y)

        for i, j in zip(range(X.shape[1]), range(len(sub_ds))):
            d = round(sub_ds[j], 2)
            raw_virtual_points[d] = {}
            raw_virtual_points[d]['X'] = X[:, i]
            raw_virtual_points[d]['Y'] = Y[:, i]
        return raw_virtual_points











    def reset(self):
        self.str_d = {}
        self.curves = {}


    def write_curve_files(self, curves):
        for c in curves:
            c = int(c)
            if not os.path.isdir(self.path_fin):
                os.makedirs(self.path_fin)
            np.savetxt(os.path.join(self.path_fin, f'{c}mm_raw-data.csv'), self.curves[c]['raw_data'], delimiter=',')


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



# TODO: implement a robust curve- / thickness-chose-mechanism
def plot(path_map, excl_filter=None):
    if not os.path.exists(os.path.join(path_map, '../visual')):
        os.mkdir(os.path.join(path_map, '../visual'))
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
            plt.savefig(os.path.join(os.path.join(path_map, '../visual'), f'SNR_T_{filename}mm_{max_kv}maxkV.png'))


def write_data(path_T, path_SNR, path_fin):
    now = time.strftime('%c')
    if not os.path.exists(os.path.join(path_fin, '../visual')):
        os.makedirs(os.path.join(path_fin, '../visual'))
    with open(os.path.join(path_fin, '../visual', 'evaluation.txt'), 'w+') as f:
        f.write(f'{now}\n')
        f.write(f'used transmission data: {path_T}\n')
        f.write(f'used SNR data: {path_SNR}\n')
        f.close()
