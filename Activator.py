import gc

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
        self.fit_kVT_x = []
        self.fit_kVT_y = []



class Activator:
    def __init__(self, data_T: list, path_db: str, U0: int):
        self.data_T = data_T
        self.path_db = path_db
        self.curves = []
        self.U0 = U0
        self.kV_interpolation = False
        self.ds = [0, 1, 4, 5, 8, 9]
        self.kVs = [40, 60, 80, 100, 120, 140, 160, 180]

    def __call__(self, *args, **kwargs):
        self.load_db()
        self.interpolate_SNR_T()
        for i in self.kVs:
            if self.U0 == i:
                print('no SNR(T) interpolation needed.')
                self.interpolate_kVT()
            else:
                self.kV_interpolation = True
        if self.kV_interpolation:
            self.interpolate_kVT()

    def load_db(self):
        db = DB(self.path_db)
        for d in self.ds:
            kV, T, SNR = db.read_data(d)                        # read curve
            self.curves.append(Curve(d=d, kV=kV, T=T, SNR=SNR))


    def interpolate_SNR_T(self):
        # 1)    interpolate SNR(T) curve in the given range of nearest neighbours
        # 2)    check the 'distance from U0 to the lower neighbour dist = self.U0 - l_neighbour
        for _curve in self.curves:
            dist, lb, rb = self.find_neighbours()
            l_idx = self.kVs.index(lb)                      # get index of kV to get range of interpolation
            r_idx = self.kVs.index(rb)
            step = rb - lb
            step = complex(str(step) + 'j')                    # convert int(step) to string to get the imagenary notation for indexing
            f = interpolate.interp1d(_curve.T, _curve.SNR, kind='linear')
            x = np.r_[_curve.T[l_idx]:_curve.T[r_idx]:step]
            _curve.fit_SNRT_y = f(x)
            _curve.fit_SNRT_x = np.r_[_curve.T[l_idx]:_curve.T[r_idx]:step]

            _curve.fit_kVT_x.append()




    def find_neighbours(self):
        num = next(i[0] for i in enumerate(self.kVs) if i[1] >= self.U0)
        neighbour_r = self.kVs[num]
        neighbour_l = self.kVs[num-1]
        dist = self.U0 - neighbour_l
        return dist, neighbour_l, neighbour_r


    def interpolate_kVT(self):
        # 1)    take the distance and add it to the lower neighbour for calculated SNR(T) @ U0

        dist, lb, rb = self.find_neighbours()

        for i in range(lb, rb):
            for _curve in self.curves:
                self.curves[i].kVT_x.append(_curve.T[i])
                self.curves[i].kVT_y.append(_curve.SNR[i])
        print('test')



    def get_db_data(self):
        pass