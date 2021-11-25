import os
import gc
import csv
import numpy as np
from scipy import interpolate
from scipy.ndimage import median_filter
from matplotlib import pyplot as plt

from externe_files import file
from Plots.Plotter import Plotter as PLT
from Plots.Plotter import TerminalColor as PCOL
from SNR_Calculation.map_generator import SNRMapGenerator
import helpers as h



def fast_CT():
    img_shape = (1536, 1944)
    header = 2048
    px_map = h.load_bad_pixel_map()

    imgs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\test_fast_CT'
    darks = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\darks'
    refs = r'\\132.187.193.8\junk\sgrischagin\2021-10-04_Sergej_Res-phantom_135kV_1mean_texp195_15proj-per-angle\refs'

    darks = file.volume.Reader(darks, mode='raw', shape=img_shape, header=header, dtype='<u2').load_all()
    refs = file.volume.Reader(refs, mode='raw', shape=img_shape, header=header, dtype='<u2').load_all()
    darks_avg = np.nanmean(darks, axis=0)
    refs_avg = np.nanmean(refs, axis=0)

    list_Ts = []
    list_angles = []
    data = file.volume.Reader(imgs, mode='raw', shape=img_shape, header=header, dtype='<u2').load_all()
    data_avg = np.nanmean(data, axis=0).astype(data.dtype, copy=False)

    medfilt_image = median_filter(data_avg, 5, mode='nearest')[px_map]
    for k in range(len(data)):
        data[k][px_map] = medfilt_image

    for i in range(len(data)):
        if darks is None and refs is None:
            T = data[np.where(data > 0)].min()
        else:
            img = (data[i] - darks_avg) / (refs_avg - darks_avg)
            T = img[np.where(img > 0)].min()
        list_Ts.append(T)
        list_angles.append(i)
    a = np.asarray(list_Ts)
    b = np.asarray(list_angles)
    del img, data, refs, darks
    gc.collect()
    return np.vstack([a, b])


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
                    d = h.extract_d(file)
                    if f'_{d}mm_' in file or f'_{d}-mm_' or f'_{d}_mm_':
                        loc.append(os.path.join(_subdir, file))
        self.files['SNR'] = loc

    def collect_transmission_files(self):
        loc_fs = []
        loc_ds = []
        for file in os.listdir(self.p_T_files):
            if file.endswith('.csv'):
                d = h.extract_d(file)
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


class Activator():
    def __init__(self, snr_files: str, T_files: str, U0: int, snr_user: float, kv_ex: list = None, ds: list = None,
                 ssize=None, vir_curve_step: float = None, create_plot: bool = False):
        self.fast_CT_data = None
        self.T_min = None

        self.kv_ex = kv_ex

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
        if self.ds is None:
            self.ds = self.scanner.files['ds']

        self.x_U0_c = []
        self.y_U0_c = []
        self.x_U0_points = []
        self.y_U0_points = []

        if vir_curve_step is None:
            self.vir_curve_step = 0.1
        else:
            self.vir_curve_step = vir_curve_step
        self.intercept = {'x': {}, 'y': {}}
        self.intercept_found = False
        self.d_opt = None
        self.X_opt = None
        self.Y_opt = None
        self.U_opt = {'val': None, 'fit': {}, 'data': {}}
        self.opt_data_points = None
        self.map = None
        self.U_0 = {'val': self.U0, 'fit': {}, 'data': {}}
        self.Generator = SNRMapGenerator(scanner=self.scanner, d=self.ds, kV_filter=kv_ex)

    def __call__(self, create_plot: bool = True, *args, **kwargs):

        #self.fast_CT_data = fast_CT()
        self.fast_CT_data = [[0.513, 0.255, 0.319, 0.419, 0.351, 0.359, 0.473], [0.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0]]
        self.T_min = self.get_min_T()

        self.map = self.Generator(spatial_range=self.ssize)
        self.map['T_min'] = self.T_min
        self.map['ds'] = self.ds
        self.create_virtual_curves()

        self.U_0['fit'], self.U_0['data'] = self.create_monoKV_curve(kV_val=self.U0)
        self.map['U0_curve'] = self.U_0
        self.d_curve_T_intercept(curve=self.U_0['fit'])

        if self.intercept_found:
            self.get_opt_SNR_curve()
            self.map['opt_curve'] = self.U_opt
            self.map['intercept_found'] = True
        else:
            self.map['intercept_found'] = False

        self.calc_t_exp()
        self.printer()
        if create_plot:
            _plt = PLT()
            _plt.create_v_plot(path_result=self.scanner.path_fin, object=self.map, full=True)


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


    def d_curve_T_intercept(self, curve):
        # TODO: more robust idx calculation. Catching cases like 1 < len(idx). ->
        epsilons = [0.000001, 0.00001, 0.0001]
        for eps in epsilons:
            idx = np.where((curve[:, 0] > (self.T_min - eps)) & (curve[:, 0] < (self.T_min + eps)))
            if len(idx[0]) > 1:
                # if len(idx) > 1 than the central value where the condition holds, will be selected
                mid_idx = int(abs(idx[0][-1] - idx[0][0]) / 2)
                idx = idx[0][mid_idx]
            else:
                idx = np.where((curve[:, 0] > (self.T_min - eps)) & (curve[:, 0] < (self.T_min + eps)))[0][0]
            if idx.size:
                self.intercept['x'] = self.T_min
                self.intercept['y'] = curve[:, 1][idx]
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

        temp_curves = {}

        # TODO: zuerst sortieren udn dann d, i in zip stuff machen.

        for d, i in zip(self.map['d_curves'], range(len(self.ds)-1)):
            X = []
            Y = []

            ds = list(self.map['d_curves'])
            c_num = np.arange(ds[i], ds[i + 1], self.vir_curve_step)[1:]
            c_num = c_num[::-1]

            d2 = ds[i + 1]
            d1 = ds[i]
            _c2 = self.map['d_curves'][d2]['data']
            _c1 = self.map['d_curves'][d1]['data']
            kV_2, T_2, SNR_2 = _c2[:, 0], _c2[:, 1], _c2[:, 2]
            kV_1, T_1, SNR_1 = _c1[:, 0], _c1[:, 1], _c1[:, 2]


            #       CREATING EQUALLY SPACED VERTICAL 'VIRTUAL' DATA POINTS BETWEEN REAL DATA POINTS
            #       -> virtual curve pillows

            # 1)    pick T value from second curve (d2) and T value from first curve (d1)
            # 2)    pick SNR value from second curve and SNR value from first curve
            # 3)    create a line between T_2 and T_1 values and linspace it into len(c_num) + 2 points
            for j in range(len(T_1)):
                _x = [T_2[j], T_1[j]]
                _y = [SNR_2[j], SNR_1[j]]
                f = interpolate.interp1d(_x, _y, kind='linear')
                _x_new = np.linspace(T_2[j], T_1[j], len(c_num) + 2)[1:-1]
                _y_new = f(_x_new)
                X.append(_x_new)
                Y.append(_y_new)


            #       PICK A CURVE AND FIT IT

            # 1)    for the length of entries of the T data, which should be the same length as SNT data,
            #       append just the picked curve to the _T/_SNR array
            # 2)    fit the _T/_SNR arrays
            for k in range(c_num.size):
                _T = []
                _SNR = []
                _d = round(c_num[k], 2)
                for _j in range(len(T_1)):
                    _T.append(X[_j][k])
                    _SNR.append(Y[_j][k])

                # full scatter curve _T and _SNR
                _T = np.asarray(_T)
                _SNR = np.asarray(_SNR)

                merged_data_curve = self.Generator.merge_data(kV=kV_1, T=_T, SNR=_SNR)
                temp_curves[_d] = {}
                temp_curves[_d]['data'] = merged_data_curve

                kv_grid = h.create_kv_grid(data_curve=merged_data_curve)
                merged_grid_curve = self.Generator.merge_data(kV=kv_grid[:, 0], T=kv_grid[:, 1], SNR=kv_grid[:, 2])
                temp_curves[_d]['kv_grid'] = merged_grid_curve

        self.map['d_curves'].update(temp_curves)
        self.map['d_curves'] = dict(sorted(self.map['d_curves'].items()))

        self.find_curve_max()


    def prep_curve(self, x, y):
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2 + yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0], u])

        t = np.linspace(0, u.max(), 141)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        return xn, yn


    def find_curve_max(self):
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['kv_grid']
            idx = np.argmax(_c[:, 2])
            self.map['d_curves'][d]['max_idx'] = idx


    def make_monokV_curve(self, kV):
        # suche in jeder Kurve den index der np.where(curve == kV) ist
        X = []
        Y = []
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['kv_grid']
            idx = np.where(_c[:, 0] == kV)[0][0]
            x_val = _c[:, 1][idx]
            y_val = _c[:, 2][idx]
            X.append(x_val)
            Y.append(y_val)
        return np.vstack((X, Y)).T


    def find_neighbours(self, kV):
        # find first element in map where arg. >= self.U0, which is the right border of the searched interval
        # the nearest left entry is the left border between which the interpolation will take place
        d = None
        for d in self.map['d_curves']:
            break
        if d is not None:
            _c_kV = self.map['d_curves'][d]['data'][:, 0].tolist()

            num = next(i[0] for i in enumerate(_c_kV) if i[1] > kV)
            left_nbr = _c_kV[num-1]
            right_nbr = _c_kV[num]
            dist = kV - left_nbr

            return abs(int(dist)), int(left_nbr), int(right_nbr), (int(right_nbr) - int(left_nbr))
        else:
            print('Could not estimate \'neighbours\'.')


    def get_opt_SNR_curve(self):
        '''
        delta:   abs between curves SNR value at T_min and the intercept['y'] value
        '''
        #   1) find min abs between x value of each curve and intercept['x'] get the index of the min value
        #   2) compare the SNR value at found index of each curve -> find MIN( abs(SNR_val[idx] - intercept['y']) )

        old_delta = None
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['kv_grid']

            #   1) estimate the nearest interpolated x values to the intercept_x
            idx = (np.abs(_c[:, 1] - self.T_min)).argmin()

            delta = abs(_c[:, 2][idx] - self.intercept['y'])
            if old_delta is None:
                old_delta = delta
            elif delta < old_delta:
                old_delta = delta
                self.map['d_opt'] = d

        self.pick_opt_curve()


    def filter_relevant_curves(self):
        watch_curves = {}
        for d in self.map['d_curves']:
            _c_SNR = self.map['d_curves'][d]['fit'][:, 2]
            if not max(_c_SNR) < self.intercept['y']:
                watch_curves[d] = self.map['d_curves'][d]
        return watch_curves


    def pick_opt_curve(self):
        _c = self.map['d_curves'][self.map['d_opt']]['kv_grid']
        _c_kV = _c[:, 0]
        _c_T = _c[:, 1]
        _c_SNR = _c[:, 2]
        idx = np.argmax(_c_SNR)

        self.U_opt['val'] = _c_kV[idx]
        self.U_opt['d'] = self.map['d_opt']
        self.U_opt['fit'], self.U_opt['data'] = self.create_monoKV_curve(kV_val=_c_kV[idx])


    def search_nearest_curve(self):
        # 1) accept transmission array from fast_ct() t_arr = [[p1, p2,..], [T1, T2 ..]]
        # 2) translate from T(proj) to d. There is a need in continuous d values
        pass


    def create_monoKV_curve(self, kV_val):
        monokV_points = self.make_monokV_curve(kV=kV_val)
        curve_fit, data_points = self.vertical_interpolation(points=monokV_points)
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
        if self.intercept_found:
            icpt_x = self.intercept['x']
            icpt_y = self.intercept['y']
            d_opt = self.map['d_opt']
            kV_opt = self.U_opt['val']
            print(f'intercept T_min and U0:\n' f'({round(icpt_x, 3)} / {round(icpt_y, 3)})')
            print(' ')
            print(f'interpolated thickness at intercept (d_opt):\n'  f'{d_opt}')
            print(' ')
            print('optimal voltage for measurement:\n' + f' ==> kV_opt = {kV_opt} kV <==' + '\n')
        else:
            print('No intercept between U0 and T_min could be found. \n'
                  '-> You may reduce epsilon in find_intercept()')


    def poly_fit(self, var_x, var_y, steps):
        a, b, c = np.polyfit(var_x, var_y, deg=2)
        x = np.linspace(var_x[0], var_x[-1], steps)
        y = self.func_poly(x, a, b, c)
        return x, y


    def overwatch_plot(self):
        fig = plt.figure()
        ax = fig.add_subplot()

        ds = [1.0, 1.1, 3.9, 4.0]
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['fit']
            if d in ds:
                ax.plot(_c[:, 1], _c[:, 2], label=f'{d} mm')
            else:
                pass

        plt.legend()
        plt.show()


    @staticmethod
    def func_poly(x, a, b, c):
        return a * x ** 2 + b * x + c
