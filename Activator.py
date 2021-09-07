import gc

from SNR_Calculation.CurveDB import *
import numpy.polynomial.polynomial as poly
from scipy import interpolate
from scipy.optimize import fsolve
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
        self.U0_curve_x = []
        self.U0_curve_y = []
        self.idx = None


    def __call__(self, *args, **kwargs):
        #while not self.stop_exe:
        self.load_db()
        self.interpolate_SNR_T()
        self.interpolate_U0()
        #self.idx = self.find_intercept()
        self.create_plot()

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
        #dist, lb, rb = self.find_neighbours()
        step = self.kVs[-1] - self.kVs[0]
        idx = self.U0 - self.kVs[0]
        for _curve in self.curves:
            f = interpolate.interp1d(_curve.T, _curve.SNR, kind='linear')
            self.x_U0 = np.linspace(_curve.T[0], _curve.T[-1], step + 1)
            self.f_U0 = f(self.x_U0)
            self.U0_curve_x.append(self.x_U0[idx])
            self.U0_curve_y.append(self.f_U0[idx])


    def interpolate_U0(self):
        self.f_U0 = interpolate.interp1d(self.U0_curve_x, self.U0_curve_y, kind='linear')
        self.x_U0 = np.linspace(self.U0_curve_x[0], self.U0_curve_x[-1], 141)
        #self.x_U0 = np.r_[x[0]:x[-1]:100j]


    def find_intercept(self):


        num = len(self.U0_curve_y)
        f_Tmin = np.full((num,), self.min_T)
        f_U0 = np.asarray(self.U0_curve_y)

        result = self.findIntersection(f_U0, f_Tmin, 0.0)

        idx = np.argwhere(np.diff(np.sign(f_U0 - f_Tmin))).flatten()
        return int(idx)



    def find_neighbours(self):
        # find first element in self.curves which where arg. > self.U0 ==> right border
        # the nearest left entry is the left border between which the interpolation will take place
        num = next(i[0] for i in enumerate(self.curves[0].kV) if i[1] >= self.U0)
        neighbour_l = self.curves[0].kV[num-1]
        neighbour_r = self.curves[0].kV[num]
        dist = self.U0 - neighbour_r
        return abs(int(dist)), int(neighbour_l), int(neighbour_r)

    def interpolate_kVT(self):
        dist, lb, rb = self.find_neighbours()
        for i in range(lb, rb):
            for _curve in self.curves:
                self.curves[i].kVT_x.append(_curve.T[i])
                self.curves[i].kVT_y.append(_curve.SNR[i])

    def create_plot(self):
        path_res = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\2021-8-30_Evaluation\Eval_Result'
        fig = plt.figure()
        ax = fig.add_subplot()
        for _c in self.curves:
            T = _c.T
            SNR = _c.SNR
            plt.plot(T, SNR, label=f'{_c.d}mm', marker='o', c='grey')
            plt.plot(_c.kVT_x, _c.kVT_y, c='black', marker='2', markersize='15')

            ax.text(_c.T[0]-0.05, _c.SNR[0], f'{_c.d}mm')
        #plt.plot(T[self.idx], SNR[self.idx], marker='o', c='red')
        plt.title(f'Thickness dependent $SRN(T)$ with $U_{0} = {self.U0}$kV')
        #plt.tight_layout()
        plt.xlabel('Transmission a.u.')
        plt.ylabel('SNR')
        #plt.legend()
        plt.plot(self.U0_curve_x, self.U0_curve_y, c='blue', linestyle='-')
        plt.axvline(x=self.min_T, c='blue', linestyle='-')
        plt.show()
        fig.savefig(os.path.join(path_res, 'MAP_U0.pdf'), dpi=600)