import gc
import math
import matplotlib.pyplot as plt
from SNR_Calculation.CurveDB import *
import numpy.polynomial.polynomial as poly
from scipy import interpolate
import numpy as np


class Scanner:
    def __init__(self):
        pass


class Curve:
    def __init__(self, d, kV: list, T: list, SNR: list):
        self.d = d
        self.kV = kV
        self.T = T
        self.SNR = SNR


    def append_data(self, kV, T, SNR):
        self.kV.append(kV)
        self.T.append(T)
        self.SNR.append(SNR)


class Activator:
    def __init__(self, data_T: list, path_db: str, U0: int, ds: list):
        self.data_T = data_T
        self.min_T = self.get_min_T()
        self.path_db = path_db
        self.curves = []
        if 40 <= U0 and U0 <= 190:
            self.U0 = U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {U0} \n...exit...' )
            self.stop_exe = True
        self.kV_interpolation = False
        self.ds = ds
        self.kVs = [40, 60, 80, 100, 120, 140, 160, 180]
        self.f_U0 = []
        self.x_U0 = []
        self.x_SNR_T = []
        self.y_SNR_T = []
        self.c_U0_x = []
        self.c_U0_y = []
        self.intercept_x = 0
        self.intercept_y = 0
        self.virtual_xs = []
        self.virtual_ys = []
        self.d_opt = None
        self.X_opt = None
        self.Y_opt = None
        self.kV_opt = None


    def __call__(self, *args, **kwargs):
        self.load_db()
        self.interpolate_curve_piece()
        self.interpolate_U0()
        #self.interpolate_curves_full()
        self.create_virtual_curves()
        self.find_intercept()
        self.get_opt_SNR_curve()
        self.plot_MAP()

    def get_min_T(self):
        return np.min(self.data_T[1])

    def load_db(self):
        db = DB(self.path_db)
        for d in self.ds:
            kV, T, SNR = db.read_data(d, mode='raw')                        # read curve
            self.curves.append(Curve(d=d, kV=kV, T=T, SNR=SNR))

    def interpolate_curve_piece(self):
        # 1)    interpolate SNR(T) curve in the given range of nearest neighbours
        # 2)    check the 'distance from U0 to the lower neighbour dist = self.U0 - l_neighbour
        dist, lb, rb = self.find_neighbours()
        step = rb - lb
        for _curve in self.curves:
            # 1)    get left and right neighbours and interpolate between these values
            # 2)    append the values at given index to the c_U0x and c_U0_y
            il = _curve.kV.index(lb)
            ir = _curve.kV.index(rb)
            a, b, c = np.polyfit(_curve.T, _curve.SNR, deg=2)
            x_SNR_T = np.linspace(_curve.T[il], _curve.T[ir], step + 1)
            y_SNR_T = self.func_poly(x_SNR_T, a, b, c)
            self.c_U0_x.append(x_SNR_T[dist])
            self.c_U0_y.append(y_SNR_T[dist])

    def interpolate_curves_full(self):
        db = DB(self.path_db)
        for _c in self.curves:
            a, b, c = np.polyfit(_c.T, _c.SNR, deg=2)
            x = np.linspace(_c.T[0], _c.T[-1], 141)
            vol = np.linspace(_c.kV[0], _c.kV[-1], 141)
            y = self.func_poly(x, a, b, c)
            for i in range(len(x)):
                db.add_data(d=_c.d, voltage=vol[i], SNR=y[i], T=x[i], mode='fit')

    def interpolate_U0(self):
        self.f_U0 = interpolate.interp1d(self.c_U0_x, self.c_U0_y, kind='linear')
        self.x_U0 = np.linspace(self.c_U0_x[0], self.c_U0_x[-1], 141)

    def find_intercept(self):
        # create dummy function U0' with high sampling rate
        # find intercept with 'high precision'
        # searching for the x value where the U0x1 is greater than U0x0 for the first time
        dummy_f_U0 = interpolate.interp1d(self.c_U0_x, self.c_U0_y, kind='linear')
        dummy_x_U0 = np.linspace(self.c_U0_x[0], self.c_U0_x[-1], 10000)
        dummy_y = dummy_f_U0(dummy_x_U0)

        for i in range(len(dummy_x_U0)):
            if self.min_T == round(dummy_x_U0[i], 4):
                self.intercept_x = dummy_x_U0[i]
                self.intercept_y = dummy_y[i]
                print(f'intercept: ({round(self.intercept_x, 3)} , {round(self.intercept_y, 3)})')
                break
        if self.intercept_x == 0 and self.intercept_y == 0:
            print('no intercept between U0 and min. T found. You may reduce the round digits at find_intercept()')

    def find_neighbours(self):
        # find first element in self.curves which where arg. > self.U0 ==> right border
        # the nearest left entry is the left border between which the interpolation will take place
        num = next(i[0] for i in enumerate(self.curves[0].kV) if i[1] >= self.U0)
        neighbour_l = self.curves[0].kV[num-1]
        neighbour_r = self.curves[0].kV[num]
        dist = self.U0 - neighbour_l
        return abs(int(dist)), int(neighbour_l), int(neighbour_r)

    def interpolate_kVT(self):
        dist, lb, rb = self.find_neighbours()
        for i in range(lb, rb):
            for _curve in self.curves:
                self.curves[i].kVT_x.append(_curve.T[i])
                self.curves[i].kVT_y.append(_curve.SNR[i])

    def create_virtual_curves(self):
        #   1) read first fitted curve from db
        #   2) read second curve from db
        #   3) calc the number of curves which needed to be created between first and second in respect to the step size
        #   4) take the first data point (SNR/kV) of the second curve and the first data point (SNR/kV) of the first curve
        #      and divide the abs between them into c_num + 1 pieces
        step = 0.1
        db = DB(self.path_db)
        for i in range(len(self.ds)-1):
            X = []
            Y = []

            c_num = np.arange(self.ds[i], self.ds[i + 1], step)[1:]
            c_num = c_num[::-1]
            V_2, T_2, SNR_2 = db.read_data(d=self.ds[i + 1], mode='raw')
            V_1, T_1, SNR_1 = db.read_data(d=self.ds[i], mode='raw')

            for j in range(len(T_1)):
                _x = [T_2[j], T_1[j]]
                _y = [SNR_2[j], SNR_1[j]]
                f = interpolate.interp1d(_x, _y, kind='linear')
                _x_new = np.linspace(T_2[j], T_1[j], len(c_num)+2)[1:-1]
                _y_new = f(_x_new)
                X.append(_x_new)
                Y.append(_y_new)

            for k in range(len(c_num)):
                _T = []
                _SNR = []
                _d = round(c_num[k], 2)
                kV = V_1
                for _j in range(len(T_1)):
                    _T.append(X[_j][k])
                    _SNR.append(Y[_j][k])
                self.curves.append(Curve(d=_d, kV=kV, T=_T, SNR=_SNR))


    def get_opt_SNR_curve(self):
        #   1) take the x and y value of T_min and find between curves are 'neighbours'
        #   2)  find min abs between x and y values of curves and intercept
        #   3)  look into every curve. If the max SNR value is smaller than the actual intercept SNR value - > next curve
        #       to find _d_max
        watch_curves = []
        for _c in self.curves:
            _SNR_max = max(_c.SNR)
            if not max(_c.SNR) < self.intercept_y:
                watch_curves.append(_c)
        del _c, _SNR_max
        gc.collect()

        old_delta = None
        for i in range(len(watch_curves)):
            _c = watch_curves[i]
            _x, _y = self.poly_fit(_c.T, _c.SNR, 10000)

            #   1) estimate the indeces which lays on the left and right side of intercept_x
            idx = np.argwhere(_x < self.intercept_x)
            #rb = np.argwhere(_x > self.intercept_x)
            idx = idx.flatten()
            idx = idx[-1]

            delta = abs(_y[idx] - self.intercept_y)
            if old_delta is None:
                old_delta = delta
            if delta < old_delta:
                old_delta = delta
                self.d_opt = _c.d

        self.find_max()

    def poly_fit(self, T, SNR, steps):
        a, b, c = np.polyfit(T, SNR, deg=2)
        x = np.linspace(T[0], T[-1], steps)
        y = self.func_poly(x, a, b, c)
        return x, y

    def find_max(self):
        for _c in self.curves:
            try:
                if _c.d == self.d_opt:
                    x, y = self.poly_fit(_c.T, _c.SNR, 141)
                    self.kV_opt = np.argmax(y)
                    self.Y_opt = y[self.kV_opt]
                    self.X_opt = x[self.kV_opt]
            except:
                print('No curve satisfies the condition _c.d==self.d_opt.')




    @staticmethod
    def func_linear(x, m, t):
        return m*x + t

    @staticmethod
    def func_poly(x, a, b, c):
        return a*x**2 + b*x + c

    @staticmethod
    def whole_num(num):
        if num - int(num) == 0:
            return True
        else:
            return False

    def plot_MAP(self):
        col_red = '#D56489'
        col_yellow = '#ECE6A6'
        col_blue = '#009D9D'
        col_green = '#41BA90'
        path_res = r'C:\Users\Rechenfuchs\PycharmProjects\SNR_Preperator_new_approach'
        db = DB(self.path_db)
        fig = plt.figure()
        ax = fig.add_subplot()
        for _c in self.curves:
            if self.whole_num(_c.d):
                data_size = 40
                ax.text(_c.T[0] - 0.05, _c.SNR[0], f'{_c.d}mm')
                _alpha=0.9
                linew = 3
            else:
                data_size = 15
                _alpha = 0.2
                linew = 1

            plt.scatter(_c.T, _c.SNR, label=f'{_c.d}mm', marker='o', alpha=_alpha, c=col_red, s=data_size)   # raw data points
            a, b, c = np.polyfit(_c.T, _c.SNR, deg=2)
            x = np.linspace(_c.T[0], _c.T[-1], 141)
            y = self.func_poly(x, a, b, c)
            plt.plot(x, y, c=col_red, alpha=_alpha, linewidth=linew)


        plt.scatter(self.X_opt, self.Y_opt, marker='x', c='black', s=50)
        plt.title(f'$SRN(T)$ with $U_{0} = {self.U0}$kV       FIT: $f(x) = a x^{2} + bx + c$')
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR/s')
        plt.xlim(self.curves[-1].T[0] - 0.05, self.curves[0].T[-1] + 0.02)
        plt.plot(self.c_U0_x, self.c_U0_y, c=col_green, linestyle='-', linewidth=2)                         # U0 curve
        plt.axvline(x=self.min_T, c=col_green, linestyle='--', linewidth=1)
        plt.scatter(self.intercept_x, self.intercept_y, c=col_red, marker='x', s=50)
        plt.show()
        fig.savefig(os.path.join(path_res, 'MAP_U0.pdf'), dpi=600)

    def plot_SRN_kV(self):
        path_res = r'C:\Users\Rechenfuchs\PycharmProjects\SNR_Preperator_new_approach'
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        for _c in self.curves:
            kV = _c.kV
            SNR = _c.SNR
            ax1.plot(kV, SNR, label=f'{_c.d}mm', marker='o')
            ax2.plot(np.arange(0.0, 0.6, len(kV)), np.ones(100))
            ax2.cla()
            ax2.set_xlabel(r'T')
        ax1.set_xlabel('Voltage $[kV]$')
        ax2.set_xlabel('T')
        ax1.set_ylabel('SNR/s')
        plt.legend()
        plt.show()
        fig.savefig(os.path.join(path_res, 'SNR_kV_mod.pdf'), dpi=600)


    def fit_2D(self):
        pass