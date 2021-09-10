import time
import SNR_Calculation.CurveDB as db
import numpy as np
import os
import matplotlib.pyplot as plt


class SNRMapGenerator:
    def __init__(self, path_snr: str, path_T: str, path_fin: str, d: int, kV_filter: list = None):
        self.path_snr = path_snr
        self.path_T = path_T
        self.path_fin = path_fin

        self.kV_filter = kV_filter
        if kV_filter is not None:
            self.kV_filter = kV_filter
            print(f'You passed {self.kV_filter} as a kV filter.')
        else:
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
        self.path_db = r''
        self.db = db.DB(self.path_db)
        self._collect_data()
        self.get_T_data()
        self.get_SNR_data()
        self._merge_data()
        self.write_data()

    # TODO: implement more robust file finding routine
    def _collect_data(self):
        for file in os.listdir(self.path_snr):
            if f'_{self.d}_mm' in file and file.endswith('.txt'):
                self.txt_files.append(os.path.join(self.path_snr, file))

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

    # TODO: find a way to calculate the SNR between 150 and 250. At the moment just one value is used for 'mean' because no value fits the condition [150:250] naturally
    def _calc_data(self, file):
        l_bound = 150.0
        u_bound = 250.0
        self.int_kV = self.get_properties(file)
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
            os.makedirs(self.path_fin)
        np.savetxt(os.path.join(self.path_fin, f'{self.d_mm}.csv'), self.d_curve, delimiter=',', encoding='utf-8')

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
