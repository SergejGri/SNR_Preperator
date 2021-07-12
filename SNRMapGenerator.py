import time
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

    def __call__(self, *args, **kwargs):
        self.collect_data()
        self.get_T_data()
        self.get_SNR_data()
        self.merge_data()
        self.write_data()

    # TODO: implement more robust file finding routine
    def collect_data(self):
        working_dir = None
        for subdir in os.listdir(self.path_snr):
            if self.kV_filter != None:
                if subdir not in self.kV_filter:
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
        list_kV = []
        list_SNR = []
        list_tot = []
        index = []
        for file in self.txt_files:
            if f'_{self.d}mm_' in file:
                filename, self.str_kV, self.int_kV = self.get_properties(file)
                if self.int_kV <= self.max_kV:
                    data = np.genfromtxt(file, skip_header=3)
                    data_u = data[:, 0]
                    data_x = 1 / (2 * data_u)
                    l_bound = int(np.argwhere(data_x <= 250.0)[0])
                    u_bound = int(np.argwhere(data_x >= 150.0)[-1])
                    data_x = data_x[l_bound:u_bound+1]


                    for i in range(len(data_x)):
                        if 150 <= data_x[i] <= 250:
                            index.append(i)

                    data_SNR = data[index[0]:index[-1]+1, 1]
                    self.mean_SNR = data_SNR.mean()
                    list_kV.append(self.int_kV)
                    list_SNR.append(self.mean_SNR)
        list_tot.append(list_kV)
        list_tot.append(list_SNR)
        arr = np.asarray(list_tot).T
        self.data_SNR = arr[arr[:, 0].argsort()]
        if self.max_kV is not None:
            self.data_SNR = self.data_SNR[:self.idx[0] + 1, :]

    def merge_data(self):
        self.d_curve = np.hstack((self.data_T, self.data_SNR))
        self.d_curve = np.delete(self.d_curve, 2, axis=1)
        self.d_curve.astype(float)

    def write_data(self):
        if not os.path.exists(self.path_fin):
            os.mkdir(self.path_fin)
        np.savetxt(os.path.join(self.path_fin, f'{self.d_mm}.csv'), self.d_curve, delimiter=',', encoding='utf-8')

    @staticmethod
    def get_properties(file):
        filename = os.path.basename(file)
        str_kV = filename.split('_')[1]
        int_kV = int(str_kV.split('k')[0])
        return filename, str_kV, int_kV
    

# TODO: implement more robust curve- / thickness-chose-mechanism
def plot(path_map, excl_filter=None):
    if not os.path.exists(os.path.join(path_map, 'Plots')):
        os.mkdir(os.path.join(path_map, 'Plots'))
    for file in os.listdir(path_map):
        #if excl_filter is not None:
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


def get_d():
    thick_0 = [4, 8, 12, 16, 20, 24, 28, 32]
    thick_1 = [5, 9, 13, 17, 21, 25, 29, 33]
    thick_2 = [6, 10, 14, 18, 22, 26, 30, 34]
    thicknesses = [thick_0, thick_1, thick_2]
    return thicknesses


def main():
    base_path_fin = r'.'
    base_path_snr = r'.'
    base_path_T = r'.'
    
    # TODO: implement more robust detection of voltages/thicknesses independent on style of passed strings
    # '160kV == '160_kV' == '160' == '160kv'... // '6mm' == '6_mm' ...
    kV_filter = ['40_kV', '160_kV']
    d_filter = ['6', '16']


    thicknesses = get_d()
    for i in range(len(thicknesses)):
        for j in range(len(thicknesses[0])):
            _d = thicknesses[i][j]
            if _d not in d_filter:
                generator = SNRMapGenerator(path_snr=base_path_snr,
                                            path_T=base_path_T,
                                            path_fin=base_path_fin,
                                            d=_d,
                                            kV_filter=kV_filter)
                generator()


if __name__ == '__main__':
    main()
