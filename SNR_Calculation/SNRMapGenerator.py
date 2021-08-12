import time
import SNR_Calculation.curve_db as db
import numpy as np
import os
import matplotlib.pyplot as plt


class SNRMapGenerator:
    def __init__(self, path_snr: str, path_T: str, path_fin: str, d: int, kV_filter: list = None):
        self.path_snr = path_snr
        self.path_T = path_T
        if path_fin is not None:
            self.path_fin = path_fin
        else:
            now = time.localtime()
            self.path_fin = os.path.join(os.environ['HOMEPATH'], 'Desktop',
                                         f'SNR_Map_{now.tm_year}-{now.tm_mon}-{now.tm_mday}_'
                                         f'{now.tm_hour}-{now.tm_min}-{now.tm_sec}')

        if kV_filter is not None:
            self.kV_filter = kV_filter
            print(f'You passed {self.kV_filter} as a kV filter.')
        else:
            self.kV_filter = None
            print(f'\n'
                  f'No value for kV_filter was passed. All voltage folders are being included for evaluation.')

        self.mean_SNR = None
        self.d = d
        self.d_mm = f'{self.d}_mm'
        self.txt_files = []
        self.data_T = []
        self.idx = None
        self.data_SNR = None
        self.d_curve = None
        self.list_kV = []
        self.list_SNR = []

    def __call__(self, *args, **kwargs):
        self.db = db.DB(self.d)
        self._collect_data()
        self.get_T_data()
        self.get_SNR_data()
        self._merge_data()
        self.write_data()

    # TODO: implement more robust file finding routine
    def _collect_data(self):
        for subdir in os.listdir(self.path_snr):
            #if self.kV_filter is not None:
            #    if subdir not in self.kV_filter:
            for subsubdir in os.listdir(os.path.join(self.path_snr, subdir)):
                working_dir = os.path.join(os.path.join(self.path_snr, subdir, subsubdir))
                for file in os.listdir(working_dir):
                    working_file = os.path.join(working_dir, file)
                    if f'_{self.d}_mm' in working_file \
                            and os.path.isfile(working_file) \
                            and working_file.endswith('.txt'):
                        self.txt_files.append(working_file)

    def find_file(self, file):
        pass
        #return key_word

    def get_T_data(self):
        data_T = np.genfromtxt(os.path.join(self.path_T, f'{self.d_mm}.csv'), delimiter=';')
        data_T = data_T[data_T[:, 0].argsort()]
        self.data_T.append(data_T[:, 0])
        self.data_T.append(data_T[:, 1])
        self.data_T = np.asarray(self.data_T).T
        if self.kV_filter is not None:
            for v in self.kV_filter:
                val = float(v.split('_')[0])
                self.data_T = self.data_T[self.data_T[:, 0] != val]

    def get_SNR_data(self):
        list_tot = []
        for file in self.txt_files:
            self._calc_data(file)
        list_tot.append(self.list_kV)
        list_tot.append(self.list_SNR)
        arr = np.asarray(list_tot).T
        self.data_SNR = arr[arr[:, 0].argsort()]

    def _calc_data(self, file):
        l_bound = 150.0
        u_bound = 250.0
        filename, int_filename, self.str_kV, self.int_kV = self.get_properties(file)
        data = np.genfromtxt(file, skip_header=3)
        data_u = data[:, 0]
        data_x = 1 / (2 * data_u)
        data = np.c_[data, data_x]
        data = data[np.logical_and(data[:, 4] >= l_bound, data[:, 4] <= u_bound)]
        data_SNR = data[:, 1]
        self.mean_SNR = data_SNR.mean()
        self.list_kV.append(self.int_kV)
        self.list_SNR.append(self.mean_SNR)

    def _merge_data(self):
        self.d_curve = np.hstack((self.data_T, self.data_SNR))
        self.d_curve = np.delete(self.d_curve, 2, axis=1)
        self.d_curve.astype(float)

    def write_data(self):
        if not os.path.exists(self.path_fin):
            os.mkdir(self.path_fin)
        np.savetxt(os.path.join(self.path_fin, f'{self.d_mm}.csv'), self.d_curve, delimiter=',', encoding='utf-8')

    @staticmethod
    def get_properties(file):
        str_kV = None
        int_kV = None
        int_filename = None
        filename = os.path.splitext(file)[0]
        try:
            int_filename = int(filename.split('_')[0])
            str_kV = filename.split('_')[1]
            int_kV = int(str_kV.split('k')[0])
        except ValueError:
            pass
        return filename, int_filename, str_kV, int_kV


class Activator:
    def __init__(self, path_base: str):
        self.path_base = path_base
        self.curves = {}
        #self.d = d
        #self.d_mm = f'{self.d} mm'
        self.read_files()

    def read_files(self):
        for file in os.listdir(self.path_base):
            if os.path.isfile(os.path.join(self.path_base, file)) and file.endswith('.csv'):
                filename, int_filename, _, _ = SNRMapGenerator.get_properties(file)
                curve = np.genfromtxt(os.path.join(self.path_base, f'{filename}.csv'), delimiter=',')
                if int_filename is not None:
                    self.curves[f'{int_filename}'] = curve
                else:
                    self.curves[f'{filename}'] = curve


# TODO: implement a robust curve- / thickness-chose-mechanism
def plot(path_map, excl_filter=None):
    if not os.path.exists(os.path.join(path_map, 'Plots')):
        os.mkdir(os.path.join(path_map, 'Plots'))
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
            plt.savefig(os.path.join(os.path.join(path_map, 'Plots'), f'SNR_T_{filename}mm_{max_kv}maxkV.png'))


def write_data(path_T, path_SNR, path_fin):
    now = time.strftime('%c')
    if not os.path.exists(os.path.join(path_fin, 'Plots')):
        os.makedirs(os.path.join(path_fin, 'Plots'))
    with open(os.path.join(path_fin, 'Plots', 'evaluation.txt'), 'w+') as f:
        f.write(f'{now}\n')
        f.write(f'used transmission data: {path_T}\n')
        f.write(f'used SNR data: {path_SNR}\n')
        f.close()