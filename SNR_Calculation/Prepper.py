import gc
import time
from SNR_Calculation.CurveDB import *
import numpy as np
import os
import matplotlib.pyplot as plt


class SNRMapGenerator:
    def __init__(self, path_snr: str, path_T: str, path_fin: str, d: list, kV_filter: list = None):
        """
        :param path_snr:
        :param path_T:
        :param path_fin:
        :param d:
        :param kV_filter:
        """
        self.path_snr = path_snr
        self.path_T = path_T
        self.path_fin = path_fin

        self.ds = d
        self.str_d = None
        self.d_txt_files = {}
        self.curves = {}
        self.ROI = {}
        self.MAP_object = {}

        self.kV_filter = kV_filter
        if kV_filter is not None:
            self.kV_filter = kV_filter
            print(f'You passed {self.kV_filter} as a kV filter.')
        else:
            print(f'No value for kV_filter was passed. All voltage folders are being included for evaluation.')

        for i in range(len(self.ds)):
            self._collect_data(self.ds[i])


    def __call__(self, spatial_range, *args, **kwargs):
        lb, rb = self.check_input(spatial_range)

        for i in range(len(self.ds)):
            self.str_d = f'{self.ds[i]}_mm'
            self.curves[f'{self.str_d}'] = {}
            self.curves[f'{self.str_d}']['kV'], self.curves[f'{self.str_d}']['T'] = self.get_T_data()
            self.curves[f'{self.str_d}']['SNR'] = self.get_SNR_data(lb, rb)
            #self.merge_data(self.curves[f'{self.str_d}']['kV'],
            #                self.curves[f'{self.str_d}']['T'],
            #                self.curves[f'{self.str_d}']['SNR'])


    # TODO: implement more robust file finding routine
    def _collect_data(self, _d):
        _loc_list = []
        for _dir in os.listdir(self.path_snr):
            _subdir = os.path.join(self.path_snr, _dir)
            for file in os.listdir(_subdir):
                if file.endswith('.txt') and f'_{_d}_mm' in file:
                    _loc_list.append(os.path.join(_subdir, file))
        self.d_txt_files[f'{_d}_mm'] = _loc_list


    def create_MAP(self, spatial_range):
        if bool(self.curves):
            self.reset()
        lb, rb = self.check_input(spatial_range)

        for i in range(len(self.ds)):
            self.str_d = f'{self.ds[i]}_mm'
            self.curves[f'{self.str_d}'] = {}
            self.curves[f'{self.str_d}']['kV'], self.curves[f'{self.str_d}']['T'] = self.get_T_data()
            self.curves[f'{self.str_d}']['SNR'] = self.get_SNR_data(lb, rb)
        self.MAP_object['curves'] = self.curves
        self.MAP_object['ROIs'] = self.ROI
        return self.MAP_object

    def get_T_data(self):
        data_T = np.genfromtxt(os.path.join(self.path_T, f'{self.str_d}.csv'), delimiter=';')
        data_T = data_T[data_T[:, 0].argsort()]
        #loc_data_T.append(data_T[:, 0])
        #_data_T.append(data_T[:, 1])
        data_T = np.asarray(data_T)
        return data_T[:, 0].T, data_T[:, 1].T
        #self.data_T = np.asarray(self.data_T).T
        #if self.kV_filter is not None:
        #    for v in self.kV_filter:
        #        val = float(v.split('_')[0])
        #        self.data_T = self.data_T[self.data_T[:, 0] != val]


    def reset(self):
        self.str_d = {}
        self.curves = {}

    def get_SNR_data(self, lb, rb):
        kvs = []
        snr_means = []

        for file in self.d_txt_files[f'{self.str_d}']:
            kV, mean_SNR = self.calc_avg_SNR(file, lb, rb)
            kvs.append(kV)
            snr_means.append(mean_SNR)

        kv_arr = np.asarray(kvs).T
        snr_arr = np.asarray(snr_means).T
        arr = np.vstack((kv_arr, snr_arr)).T
        arr = arr[arr[:, 0].argsort()]
        return arr[:, 1]


    def calc_avg_SNR(self, file, lb, rb):
        int_kV = self.get_properties(file)
        data = np.genfromtxt(file, skip_header=3)
        data = self.interpolate_data(data)  # interpolate data between first and second row (there are usually no data points in the range of interest)

        data_u = data[:, 0]
        data_x = 1 / (2 * data_u)
        data = np.c_[data, data_x]
        data = data[np.logical_and(data[:, 4] >= lb, data[:, 4] <= rb)]
        mean_SNR = data[:, 1].mean()
        return int_kV, mean_SNR


    def merge_data(self,kV, T, SNR):
        zzz = np.column_stack((kV, T, SNR))
        self.d_curve = np.hstack((T, SNR))
        self.d_curve = np.delete(self.d_curve, 2, axis=1)
        self.d_curve.astype(float)
        self.curves[f'{self.str_d}mm'] = self.d_curve


    def interpolate_data(self, data, idx:tuple=None):
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


    def units_converter(self, val):
        """
        :param:
        """
        val = val * 10 ** (-6)
        return val


    def check_input(self, port):
        if isinstance(port, int):
            port = (port,)[0]
            lb = port - 0.1 * port
            rb = port + 0.1 * port
        else:
            lb = port[0]
            rb = port[1]
        self.ROI['lb'] = lb
        self.ROI['rb'] = rb
        return lb, rb

    @staticmethod
    def get_properties(file):
        str_kV = None
        int_kV = None
        filename = os.path.basename(file)
        try:
            str_kV = filename.split('kV')[0]
            int_kV = int(str_kV.split('_')[1])
        except ValueError:
            print('check naming convention of your passed files.')
            pass
        return int_kV


    '''def write_data(self):
        if not os.path.exists(self.path_fin):
            os.makedirs(self.path_fin)
        np.savetxt(os.path.join(self.path_fin, f'{self.d_mm}.csv'), self.d_curve, delimiter=',', encoding='utf-8')

    def write_data_to_DB(self):
        db = DB(self.path_db)
        for file in os.listdir(self.path_db):
            if file.endswith('.csv') or file.endswith('.CSV'):
                working_file = os.path.join(self.path_db, file)
                d = int(file.split('_mm')[0])
                with open(working_file) as f:
                    content = f.readlines()
                    content = [x.strip() for x in content]
                    for _c in range(len(content)):
                        line = content[_c]
                        kV = float(line.split(',')[0])
                        T = float(line.split(',')[1])
                        SNR = float(line.split(',')[2])
                        db.add_data(d, voltage=kV, T=T, SNR=SNR, mode='raw')'''


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
