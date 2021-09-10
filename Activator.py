import gc
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
        self.idx = None


    def __call__(self, *args, **kwargs):
        #while not self.stop_exe:
        self.load_db()
        self.interpolate_SNR_T()
        self.interpolate_U0()
        self.interpolate_curves()
        self.intercept_idx = self.find_intercept()
        self.plot_MAP()
        #self.plot_SRN_kV()

    def load_db(self):
        db = DB(self.path_db)
        for d in self.ds:
            kV, T, SNR = db.read_data(d)                        # read curve
            self.curves.append(Curve(d=d, kV=kV, T=T, SNR=SNR))


    def get_min_T(self):
        return np.min(self.data_T[1])


    def interpolate_SNR_T(self):
        # 1)    interpolate SNR(T) curve in the given range of nearest neighbours
        # 2)    check the 'distance from U0 to the lower neighbour dist = self.U0 - l_neighbour
        dist, lb, rb = self.find_neighbours()
        step = rb - lb

        #idx = self.U0 - self.kVs[0]
        for _curve in self.curves:
            # 1)    get left and right neighbours and interpolate between these values
            # 2)    append the values at given index to the c_U0x and c_U0_y
            il = _curve.kV.index(lb)
            ir = _curve.kV.index(rb)
            a, b, c = np.polyfit(_curve.T, _curve.SNR, deg=2)
            #f = interpolate.interp1d(_curve.T, _curve.SNR, kind='linear')
            x_SNR_T = np.linspace(_curve.T[il], _curve.T[ir], step + 1)
            y_SNR_T = self.func_poly(x_SNR_T, a, b, c)
            self.c_U0_x.append(x_SNR_T[dist])
            self.c_U0_y.append(y_SNR_T[dist])


    def interpolate_curves(self):
        for _curve in self.curves:
            a, b, c = np.polyfit(_curve.T, _curve.SNR, deg=2)
            #f = interpolate.interp1d(_curve.T, _curve.SNR, kind='cubic')
            x = np.linspace(_curve.T[0], _curve.T[-1], 141)
            y = self.func_poly(x, a, b, c)
            #y = f(x)
            _curve.fit_SNRT_x.append(x)
            _curve.fit_SNRT_y.append(y)
            _curve.fit_SNRT_x = np.array(_curve.fit_SNRT_x)
            _curve.fit_SNRT_y = np.array(_curve.fit_SNRT_y)

    @staticmethod
    def func_linear(x, m, t):
        return m*x + t

    @staticmethod
    def func_poly(x, a, b, c):
        return a*x**2 + b*x + c

    def interpolate_U0(self):
        #m, t = np.polyfit(self.c_U0_x, self.c_U0_y, deg=1)
        self.f_U0 = interpolate.interp1d(self.c_U0_x, self.c_U0_y, kind='linear')
        self.x_U0 = np.linspace(self.c_U0_x[0], self.c_U0_x[-1], 141)

    def find_intercept(self):
        # create dummy function U0' with high sampling rate
        # find intercept with 'high precision'
        # searching for the x value where the U0x1 is greater than U0x0 for the first time
        intercept = []
        dummy_f_U0 = interpolate.interp1d(self.c_U0_x, self.c_U0_y, kind='linear')
        dummy_x_U0 = np.linspace(self.c_U0_x[0], self.c_U0_x[-1], 10000)
        dummy_y = dummy_f_U0(dummy_x_U0)
        idx = 0
        for i in range(len(dummy_x_U0)):
            if self.min_T == round(dummy_x_U0[i], 4):
                idx = dummy_x_U0[i]
            break
        #x_val = dummy_y.index(idx)
        #idx = np.argwhere( np.logical_and(dummy_x_U0 < self.min_T, self.min_T < dummy_x_U0))


        #x = np.linspace(_curve.T[0], _curve.T[-1], 141)




        num = len(self.c_U0_y)
        f_Tmin = np.full((num,), self.min_T)
        f_U0 = np.asarray(self.c_U0_y)
        result = self.findIntersection(f_U0, f_Tmin, 0.0)
        idx = np.argwhere(np.diff(np.sign(f_U0 - f_Tmin))).flatten()
        return int(idx)

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

    def plot_MAP(self):
        path_res = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\Eval_Result'
        fig = plt.figure()
        ax = fig.add_subplot()
        for _c in self.curves:
            plt.plot(_c.fit_SNRT_x[0], _c.fit_SNRT_y[0], c='red', alpha=0.5, linestyle='-', linewidth=3)  # fitted raw data curve
            plt.scatter(_c.T, _c.SNR, label=f'{_c.d}mm', marker='o', c='grey', s=40)              # raw data points
            ax.text(_c.T[0]-0.05, _c.SNR[0], f'{_c.d}mm')
        plt.title(f'$SRN(T)$ with $U_{0} = {self.U0}$kV       FIT: $f(x) = a x^{2} + bx + c$')
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR/s')
        plt.xlim(self.curves[-1].T[0] - 0.05, self.curves[0].T[-1] + 0.02)
        plt.plot(self.c_U0_x, self.c_U0_y, c='blue', linestyle='-', linewidth=3)                         # U0 curve
        plt.axvline(x=self.min_T, c='blue', linestyle='--', linewidth=2)
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