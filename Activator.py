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
    def __init__(self, d:int, kV: list, T: list, SNR: list):
        self.d = d
        self.kV = kV
        self.T = T
        self.SNR = SNR
        self.fit_SNRT_x = []
        self.fit_SNRT_y = []
        self.kVT_x = []
        self.kVT_y = []


class Activator:
    def __init__(self, data_T: list, path_db: str, U0: int, ds: list):
        #self.stop_exe = False
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


    def __call__(self, *args, **kwargs):
        #while not self.stop_exe:
        self.load_db()
        self.interpolate_curve_piece()
        self.interpolate_U0()
        self.interpolate_curves_full()
        self.create_virtual_curves()
        self.find_intercept()
        self.plot_MAP()
        #self.plot_SRN_kV()

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
        db = DB(self.path_db)
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
        for _curve in self.curves:
            a, b, c = np.polyfit(_curve.T, _curve.SNR, deg=2)
            x = np.linspace(_curve.T[0], _curve.T[-1], 141)
            vol = np.linspace(_curve.kV[0], _curve.kV[-1], 141)
            y = self.func_poly(x, a, b, c)
            for i in range(len(x)):
                db.add_data(d=_curve.d, voltage=vol[i], SNR=y[i], T=x[i], mode='fit')
            # Folgender Code kann eigentlich weg, erfodert jedoch eine genauere Betrachtung.
            #_curve.fit_SNRT_x.append(x)
            #_curve.fit_SNRT_y.append(y)
            #_curve.fit_SNRT_x = np.array(_curve.fit_SNRT_x)
            #_curve.fit_SNRT_y = np.array(_curve.fit_SNRT_y)

    def interpolate_U0(self):
        #m, t = np.polyfit(self.c_U0_x, self.c_U0_y, deg=1)
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
        step = 1
        new_kV = []
        db = DB(self.path_db)
        for i in range(len(self.ds)):

            c_num = np.arange(self.ds[i], self.ds[i + 1], step)[1:]
            V_2, T_2, SNR_2 = db.read_data(d=self.ds[i + 1], mode='fit')
            V_1, T_1, SNR_1 = db.read_data(d=self.ds[i], mode='fit')

            dist = math.hypot(T_2[0] - T_1[0], SNR_2[0] - SNR_1[0])

            print('test')





    '''def create_virtual_curves(self):
        step = 1
        #   1) read first fitted curve from db
        #   2) read second curve from db
        #   3) calc the number of curves which needed to be created between first and second in respect to the step size
        #   4) take the first data point (SNR/kV) of the second curve and the first data point (SNR/kV) of the first curve
        #      and divide the abs between them into c_num + 1 pieces
        new_kV = []
        db = DB(self.path_db)
        for i in range(len(self.ds)):
            c_num = np.arange(self.ds[i], self.ds[i+1], step)[1:]
            V_2, _, SNR_2 = db.read_data(d=self.ds[i+1], mode='fit')
            V_1, _, SNR_1 = db.read_data(d=self.ds[i], mode='fit')

            v_step = abs(SNR_2[0] - SNR_1[0]) / (len(c_num) + 1)
            new_curve = np.arange(SNR_2[0], SNR_1[0], v_step)[1:]
            SNR_xxx = []
            for i in range(len(SNR_1)):
                SNR_xxx.append(SNR_1[i] - v_step)

            for _d in range(len(c_num)):
                for val in range(len(SNR_1)):
                    _SNR = SNR_1[val] - v_step
                    #db.add_data(d=_d, voltage=V_1, SNR=)
            print('test')
        pass'''

    @staticmethod
    def func_linear(x, m, t):
        return m*x + t

    @staticmethod
    def func_poly(x, a, b, c):
        return a*x**2 + b*x + c

    def plot_MAP(self):
        col_red = '#D56489'
        col_yellow = '#ECE6A6'
        col_blue = '#009D9D'
        col_green = '#41BA90'
        path_res = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\Eval_Result'
        db = DB(self.path_db)
        fig = plt.figure()
        ax = fig.add_subplot()
        for _c in self.curves:
            V, T, SNR = db.read_data(_c.d, mode='fit')
            plt.plot(T, SNR, c=col_red, alpha=0.6, linestyle='-', linewidth=3)
            #plt.plot(_c.fit_SNRT_x[0], _c.fit_SNRT_y[0], c=col_red, alpha=0.6, linestyle='-', linewidth=3)  # fitted raw data curve
            plt.scatter(_c.T, _c.SNR, label=f'{_c.d}mm', marker='o', c=col_red, s=40)              # raw data points
            ax.text(_c.T[0]-0.05, _c.SNR[0], f'{_c.d}mm')
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
        path_res = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\Eval_Result'
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