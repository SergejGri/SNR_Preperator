import os
import helpers as hlp
import csv


class Scanner:
    def __init__(self, p_snr, p_T, d_ex):
        self.p_snr = p_snr
        self.p_T = p_T
        self.d_ex = d_ex
        #self.snr_files = params.paths['snr_data']
        #self.T_files = params.paths['T_data']
        #self.ds_ex = params.excluded_thicknesses
        self.path_fin = os.path.join(os.path.dirname(self.p_T), 'MAP')
        self.curves = {}
        self.files = {}
        self.collect_snr_files()
        self.collect_transmission_files()

    def collect_snr_files(self):
        loc = []
        for _dir in os.listdir(self.p_snr):
            _subdir = os.path.join(self.p_snr, _dir)
            for file in os.listdir(_subdir):
                if file.endswith('.txt'):
                    d = hlp.extract(what='d', dfile=file)
                    if d in self.d_ex:
                        pass
                    else:
                        loc.append(os.path.join(_subdir, file))
        self.files['SNR'] = loc

    def collect_transmission_files(self):
        loc_fs = []
        loc_ds = []
        for file in os.listdir(self.p_T):
            if file.endswith('.csv'):
                d = hlp.extract(what='d', dfile=file)
                if d in self.d_ex:
                    pass
                else:
                    loc_ds.append(d)
                    loc_fs.append(os.path.join(self.p_T, file))
        self.files['T'] = loc_fs
        loc_ds = sorted(loc_ds, key=lambda x: int(x))
        self.files['ds'] = loc_ds

    def collect_curve_data(self, d):
        loc_list = []
        for file in os.listdir(self.p_curves):
            if file.endswith('.csv') and f'{d}_mm' in file:
                loc_list.append(os.path.join(self.p_curves, file))
        return loc_list

    @staticmethod
    def extract_values(file):
        kV = []
        T = []
        SNR = []
        with open(file) as fp:
            reader = csv.reader(fp, delimiter=',')
            data_read = [row for row in reader]
        for i in range(len(data_read)):
            kV.append(float(data_read[i][0]))
            T.append(float(data_read[i][1]))
            SNR.append(float(data_read[i][2]))
        return kV, T, SNR