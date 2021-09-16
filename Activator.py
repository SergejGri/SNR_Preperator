import gc
import math
import matplotlib.pyplot as plt
from SNR_Calculation.CurveDB import *
from scipy import interpolate
import numpy as np
import Plotter as PLT


class Scanner:
    def __init__(self):
        pass


class Curve:
    def __init__(self, d, kV: list, T: list, SNR: list):
        self.d = d
        self.kV = kV
        self.T = T
        self.SNR = SNR


class Activator:
    def __init__(self, data_T: np.array, path_db: str, U0: int, ds: list, create_plot: bool = True):
        self.data_T = data_T
        self.T_min = self.get_min_T()
        self.path_db = path_db
        self.curves = []
        self.stop_exe = False
        if 40 <= U0 and U0 <= 190:
            self.U0 = U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {U0} \n'
                  + PLT.TerminalColor.BOLD + '...exit...' + PLT.TerminalColor.END)
            self.stop_exe = True
            return
        self.kV_interpolation = False
        self.ds = ds
        self.kVs = [40, 60, 80, 100, 120, 140, 160, 180]
        self.x_U0_c = []
        self.y_U0_c = []
        self.x_U0_p = []
        self.y_U0_p = []
        self.intercept_x = None
        self.intercept_y = None
        self.intercept_found = False
        self.d_opt = None
        self.X_opt = None
        self.Y_opt = None
        self.kV_opt = None

        self.__call__(create_plot)


    def __call__(self, create_plot, *args, **kwargs):
        self.load_db()
        self.interpolate_curve_piece()
        self.create_virtual_curves()
        self.find_intercept()
        if self.intercept_found == True:
            self.get_opt_SNR_curve()
        self.printer()

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
            self.x_U0_p.append(x_SNR_T[dist])
            self.y_U0_p.append(y_SNR_T[dist])

        self.interpolate_U0()


    def interpolate_U0(self):
        step = 10000
        f_U0 = interpolate.interp1d(self.x_U0_p, self.y_U0_p, kind='linear')
        self.x_U0_c = np.linspace(self.x_U0_p[0], self.x_U0_p[-1], step)
        self.y_U0_c = f_U0(self.x_U0_c)


    def find_intercept(self):
        # TODO: need a finally statement at the try/except block -> worst case is stopt the execution or pass 'standard values'?
        # TODO: more robust idx calculation. Catching cases like 1 < len(idx). ->
        # create dummy function U0' with high sampling rate
        # find intercept with 'high precision'
        # searching for the x value where the U0x1 is greater than U0x0 for the first time
        try:
            epsilon = 0.0005
            idx = np.where(np.logical_and(self.x_U0_c > (self.T_min - epsilon),
                                          self.x_U0_c < (self.T_min + epsilon)) == True)
            self.intercept_x = self.T_min
            self.intercept_y = self.y_U0_c[idx[0][0]]
            print(f'x:{self.intercept_x}')
            print(f'y:{self.intercept_y}')
            self.intercept_found = True
        except:
            self.stop_exe = True
            return


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


    def find_neighbours(self):
        # find first element in self.curves which where arg. > self.U0 ==> right border
        # the nearest left entry is the left border between which the interpolation will take place
        num = next(i[0] for i in enumerate(self.curves[0].kV) if i[1] >= self.U0)
        neighbour_l = self.curves[0].kV[num-1]
        neighbour_r = self.curves[0].kV[num]
        dist = self.U0 - neighbour_l
        return abs(int(dist)), int(neighbour_l), int(neighbour_r)


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
            idx = idx.flatten()
            idx = idx[-1]

            delta = abs(_y[idx] - self.intercept_y)
            if old_delta is None:
                old_delta = delta
            if delta < old_delta:
                old_delta = delta
                self.d_opt = _c.d

        self.find_max()

    def find_max(self):
        for _c in self.curves:
            try:
                if _c.d == self.d_opt:
                    x, y = self.poly_fit(_c.T, _c.SNR, 141)
                    kVx, kVy = self.poly_fit(_c.kV, _c.SNR, 141)
                    idx = np.argmax(y)
                    idxx = np.argmax(kVy)
                    self.kV_opt = kVx[idxx]
                    self.Y_opt = y[idx]
                    self.X_opt = x[idx]
                    print(f'idxx: {idxx}')
                    print(f'kV[idxx]: {kVx[idxx]}')
                    print(f'kV[0]: {_c.kV[0]}')
                    print(f'kV[-1]: {_c.kV[-1]}')
                    print(f'T[0]: {_c.T[0]}')
                    print(f'T[-1]: {_c.T[-1]}')
                    print('test')
            except:
                print('No curve satisfies the condition _c.d==self.d_opt.')


    def printer(self):
        if self.intercept_found == True:
            print(f'intercept T_min and U0:\n'
                  f'({round(self.intercept_x, 3)} / {round(self.intercept_y, 3)})')
            print(' ')
            print(f'interpolated thickness at intercept (d_opt):\n'
                  f'{self.d_opt}')
            print(' ')
            #print(f'y_max of interpolated d_opt curve (maximum SNR): {round(self.Y_opt, 3)}')
            print('optimal voltage for measurement:\n'
                  + PLT.TerminalColor.BOLD + f' ==> kV_opt = {self.kV_opt} kV <==' + PLT.TerminalColor.END + '\n')
        else:
            print('No intercept between U0 and T_min could be found. \n'
                  '1) You may reduce the round digits at find_intercept()')

    def poly_fit(self, var_x, var_y, steps):
        a, b, c = np.polyfit(var_x, var_y, deg=2)
        x = np.linspace(var_x[0], var_x[-1], steps)
        y = self.func_poly(x, a, b, c)
        return x, y

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