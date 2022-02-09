import copy
import os
import sys
import numpy as np
from scipy import interpolate

from ct_operations import CT
from ct_operations import merging_multi_img_CT
from snr_evaluator import SNREvaluator
from ext import file
from map_generator import SNRMapGenerator
from image_loader import ImageLoader
import helpers as hlp
from Plotter import Plotter as _plt
import matplotlib as mpl
import matplotlib.pyplot as plt


class Activator:

    DEBUG = True
    AVG_MAX = 4
    AVG_MIN = 1

    def __init__(self, attributes):
        self.paths = attributes.paths
        self.p_fin = os.path.join(os.path.dirname(self.paths['MAP_T_files']), 'Karte')
        self.fCT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}
        self.CT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}

        self.CT_steps = attributes.CT_steps
        self.fCT_steps = attributes.fCT_steps
        self.imgs_per_angle = attributes.imgs_per_angle
        self.T_min = None
        self.mode_avg = False
        self.MAX_CORR = 0
        if attributes.base_texp is not None:
            self.mode_avg = True
            self.BASE_texp = attributes.base_texp

        self.sub_bin = attributes.snr_bins

        self.kv_ex = []
        if attributes.excluded_kvs is not None:
            self.kv_ex = attributes.excluded_kvs
        self.ds_ex = []
        if attributes.excluded_thicknesses is not None:
            self.ds_ex = attributes.excluded_thicknesses

        if attributes.spatial_size:
            self._ssize = attributes.spatial_size
        else:
            self._ssize = (100)
            self.init_MAP = True
            print('No spatial_size value was passed: Initial MAP creation @ 100E-6 m')
        self.USR_SNR = attributes.USR_SNR

        self.stop_exe = False

        if attributes.min_kv <= attributes.U0 <= attributes.max_kv:
            self._U0 = attributes.U0
        else:
            print(f'The adjust Voltage is out of range! U0 = {attributes.U0} \n'
                  + '...exit...')
            sys.exit()

        self.kV_interpolation = False

        if attributes.virtual_curve_step is None:
            self.v_curve_step = 0.1
        else:
            self.v_curve_step = attributes.virtual_curve_step

        self.U0_intercept = {'x': {}, 'y': {}, 'd': {}}
        self.Ubest_curve = {'val': None, 'fit': {}, 'data': {}}
        self.U0_curve = {'val': self._U0, 'fit': {}, 'raw_data': {}}
        self.cumputed_ct_values = {'CT_data': None, 'fCT_data': None}

        self.view = slice(None, None), slice(50, 945), slice(866, 1040)

        self.generator = SNRMapGenerator(p_snr=attributes.paths['MAP_snr_files'],
                                         p_T=attributes.paths['MAP_T_files'],
                                         ds=attributes.ds,
                                         kv_filter=attributes.excluded_kvs)
        self._plt = _plt()


    def __call__(self, create_plot: bool = True, detailed: bool = False):
        self.evaluate_fCT(image_loader=ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False))
        self.activate_map()

        if self.map['iU0']['y'] is not None:
            self.create_Ubest_curve(d=self.map['iU0']['d'])
            self.printer()

        self.extract_data_for_fCT()
        self.reset_ct_data()
        self.evaluate_CT_no_merge()
        self.reset_ct_data()
        self.evaluate_CT_merge()        # 30 imgs for SNR

        if create_plot:
            self._plt.T_kv_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            self._plt.snr_kv_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            self._plt.map_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            #self._plt.compare_fCT_CT(object=self)



    def vertical_interpolation(self, points: np.array):
        '''


        '''
        step = 100000
        _f = interpolate.interp1d(points[:, 0], points[:, 1], kind='linear')
        x_fit = np.linspace(points[:, 0][0], points[:, 0][-1], step)
        y_fit = _f(x_fit)
        data = np.vstack((_f.x, _f.y)).T
        curve = np.vstack((x_fit, y_fit)).T
        return curve, data


    def make_monokV_curve(self, kV):
        # suche in jeder Kurve den index der np.where(curve == kV) ist
        X = []
        Y = []
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['full']
            idx = np.where(_c[:, 0] == kV)[0]
            x_val = _c[:, 1][idx]
            y_val = _c[:, 3][idx]
            X.append(x_val[0])
            Y.append(y_val[0])
        return np.vstack((X, Y)).T


    def filter_candidates(self, T_val):
        curves = {}
        for d in self.map['d_curves']:
            _c_T = self.map['d_curves'][d]['full'][:, 1]
            _c_SNR = self.map['d_curves'][d]['full'][:, 3]
            if _c_T[0] <= T_val <= _c_T[-1]:
                curves[d] = self.map['d_curves'][d]
        return curves


    def create_monoKV_curve(self, kV_val):
        monokV_points = self.make_monokV_curve(kV=kV_val)
        curve_fit, data_points = self.vertical_interpolation(points=monokV_points)
        return curve_fit, data_points[::-1]


    def make_mono_kv_curve(self, U_val):
        T = []
        SNR = []
        for d in self.map['d_curves']:
            _c = self.map['d_curves'][d]['full']
            kv = _c[:, 0]
            transmission = _c[:, 1]
            snr = _c[:, 3]

            val, idx = hlp.find_nearest(array=kv, value=U_val)
            T.append(transmission[idx])
            SNR.append(snr[idx])

        T = np.asarray(T)
        SNR = np.asarray(SNR)
        return T, SNR


    def find_intercept(self, kv_val, T_val):
        """
        find_intercept() searches for T(U0)-T_min intercept
        The function iterates through each curve and compares the difference (delta) between actual curve transmission
        value and the previous one AT given index (idx / voltage value). --> searches for min. deviation. The curve with
        minimal deviation corresponds to the searched thickness _d.

        returns T, snr, d

        :param kv_val:
        :param kv:
        """

        old_delta = None
        _d = None
        idx = None
        candidates = self.filter_candidates(T_val=T_val)

        for d in candidates:
            _c = candidates[d]['full']
            kv = _c[:, 0]
            T = _c[:, 1]

            _, idx = hlp.find_nearest(array=kv, value=kv_val)

            delta = np.abs(T[idx] - T_val).min()
            if old_delta is None:
                old_delta = delta
            elif delta < old_delta:
                old_delta = delta
                _d = d

        isnr = self.map['d_curves'][_d]['full'][:, 3][idx]
        iT = self.map['d_curves'][_d]['full'][:, 1][idx]
        return iT, isnr, _d


    def create_U0_curve(self, U0):
        self.map['U0_curve'] = {'U0_val': None, 'raw_data': None}
        T, SNR = self.make_mono_kv_curve(U_val=U0)
        self.map['U0_curve']['U0_val'] = U0
        self.map['U0_curve']['raw_data'] = hlp.merge_v1D(T, SNR)


    def create_Ubest_curve(self, d):
        _c = self.map['d_curves'][d]['full']
        kv = _c[:, 0]
        snr = _c[:, 3]
        max_val, idx = hlp.find_max(array=snr)
        kv_opt = kv[idx]

        self.map['Ubest_curve'] = {}
        T, SNR = self.make_mono_kv_curve(U_val=kv_opt)
        self.map['Ubest_curve']['Ubest_val'] = kv_opt
        self.map['Ubest_curve']['raw_data'] = hlp.merge_v1D(T, SNR)


    def get_avgs_from_fCT(self, btexp):
        fCT_texp = self.fCT_data['texp']
        loc_avgs = []
        for i in range(fCT_texp.size):
            avg_nmum = hlp.round_to_nearest(btexp, fCT_texp[i])
            loc_avgs.append(avg_nmum)
        return np.asarray(loc_avgs)


    def extract_data_for_fCT(self):
        fCT_T = np.asarray(self.fCT_data['T'])
        _, snr, d = self.extract_MAP_data(kv_val=self._U0, T=fCT_T)
        self.fCT_data['snr'] = snr
        self.fCT_data['d'] = d
        t_exp, avgs = self.calc_texp(snr=snr)
        self.fCT_data['texp'] = t_exp
        self.fCT_data['avg_num'] = avgs
        self.write_data(key='fct')


    def extract_MAP_data(self, kv_val: float, T: np.ndarray):
        '''
        searches at a given angle for snr, transmission and thickness
        :param kv_val:      voltage (curve) value at which you desire to find the intercept
        :param T:           transmission value at which you desire to find the intercept
        '''
        list_iT, list_isnr, list_id = [], [], []
        for i in range(T.size):
            iT, isnr, id = self.find_intercept(kv_val=kv_val, T_val=T[i])
            list_iT.append(iT), list_isnr.append(isnr), list_id.append(id)
        return np.asarray(list_iT), np.asarray(list_isnr), np.asarray(list_id)


    def evaluate_fCT(self, image_loader):
        imgs = image_loader.load_stack(path=self.paths['fCT_imgs'])
        refs = image_loader.load_stack(path=self.paths['fCT_refs'])
        darks = image_loader.load_stack(path=self.paths['fCT_darks'])
        imgs = imgs[self.view]
        refs = refs[self.view]
        darks = darks[self.view]
        self.fCT_data['T'], self.fCT_data['theta'] = CT(imgs=imgs, refs=refs, darks=darks, detailed=True)
        self.T_min, _ = hlp.find_min(self.fCT_data['T'])

    # 30er schritte
    def evaluate_CT_merge(self):
        new_mergings = False
        print('30er CT')
        loader_nh = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
        loader_nh.header = 0  # must be assigned additionally, since merged imgs do not contain a header
        list_images = [f for f in os.listdir(self.paths['CT_imgs']) if os.path.isfile(os.path.join(self.paths['CT_imgs'], f))]
        self.CT_data['avg_num'], self.CT_data['texp'], self.CT_data['theta'] = self.interpolate_avg_num(angles=self.CT_steps)

        if new_mergings:
            loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
            self.paths['CT_avg_imgs'] = merging_multi_img_CT(self, self.paths['CT_imgs'], list_images, self.imgs_per_angle, img_loader=loader)
            self.paths['CT_avg_imgs'] = r'\\132.187.193.8\\junk\\sgrischagin\\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\\merged-CT\\imgs'
            refs = loader.load_stack(path=self.paths['CT_refs'])
            darks = loader.load_stack(path=self.paths['CT_darks'])
            self.perform_mergings(img_stack=refs, key='refs')
            self.perform_mergings(img_stack=darks, key='darks')

        snr_evaluator = SNREvaluator(watt=5.0, voltage=102, magnification=4.45914)
        Ts = []
        snrs = []
        angles = np.arange(0, 360, 360 / self.CT_steps)

        self.MAX_CORR = 25
        for i, j in zip(range(0, len(list_images), self.sub_bin), np.arange(0, self.CT_steps)):
            # ATTENTION: projections must be loaded with image_loader where header is set to 0!
            avg = self.CT_data['avg_num'][j]
            texp = round(self.CT_data['texp'][j], 2)
            theta = round(angles[j], 1)
            imgs = loader_nh.load_stack(path=self.paths['CT_avg_imgs'], stack_range=(i, i+self.sub_bin))
            refs = loader_nh.load_stack(path=os.path.join(self.paths['CT_avg_refs'], f'avg_{avg}'), stack_range=(0, self.MAX_CORR))
            darks = loader_nh.load_stack(path=os.path.join(self.paths['CT_avg_darks'], f'avg_{avg}'), stack_range=(0, self.MAX_CORR))

            imgs = imgs[self.view]
            refs = refs[self.view]
            darks = darks[self.view]
            transmission, snr = snr_evaluator.snr_3D(generator=self.generator,
                                              result_path=self.paths['result_path'] + r'\snr_eval_30imgs',
                                              data=imgs, refs=refs, darks=darks, texp=texp, angle=theta)
            snr = snr * texp
            Ts.append(transmission)
            snrs.append(snr)
        self.CT_data['T'] = np.asarray(Ts)
        self.CT_data['snr'] = np.asarray(snrs)
        Ubest = self.map['Ubest_curve']['Ubest_val']
        _, _, self.CT_data['d'] = self.extract_MAP_data(kv_val=Ubest, T=self.CT_data['T'])
        self.write_data(key='ct30')


    def evaluate_CT_no_merge(self):
        print('120er CT')
        # 120 schritte
        T_mins = []
        snrs = []
        loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
        snr_evaluator = SNREvaluator(watt=5.0, voltage=102, magnification=4.45914)
        #list_images = [f for f in os.listdir(self.paths['CT_imgs']) if os.path.isfile(os.path.join(self.paths['CT_imgs'], f))]
        list_images = 6000
        angles = np.arange(0, 360, 360 / self.CT_steps)

        refs = loader.load_stack(path=self.paths['CT_refs'])
        darks = loader.load_stack(path=self.paths['CT_darks'])
        refs = refs[self.view]
        darks = darks[self.view]

        STEP = 120
        texp = 0.1
        j = 0
        for i in range(0, list_images, STEP):      # after debugmode change to range(0, len(list_images), STEP):
            print(f'ct120: {round( j*STEP/list_images ,2)*100} done')
            imgs = loader.load_stack(path=self.paths['CT_imgs'], stack_range=(i, i + STEP))
            imgs = imgs[self.view]
            transmission, snr = snr_evaluator.snr_3D(generator=self.generator,
                                              result_path=self.paths['result_path'] + r'\snr_eval_120imgs',
                                              data=imgs, refs=refs, darks=darks, texp=texp,
                                              angle=round(angles[j], 1))
            snr = snr * texp
            T_mins.append(transmission)
            snrs.append(snr)
            j += 1
        self.CT_data['T'] = np.asarray(T_mins)
        self.CT_data['snr'] = np.asarray(snrs)

        Ubest = self.map['Ubest_curve']['Ubest_val']
        _, _, self.CT_data['d'] = self.extract_MAP_data(kv_val=Ubest, T=self.CT_data['T'])
        _, _, self.CT_data['theta'] = self.interpolate_avg_num(angles=self.CT_steps)
        self.CT_data['texp'] = np.empty(self.CT_data['snr'].size)
        self.CT_data['texp'].fill(self.BASE_texp)
        self.CT_data['avg_num'] = np.empty(self.CT_data['snr'].size)
        self.CT_data['avg_num'].fill(1)
        self.write_data(key='ct120')


    def write_data(self, key):
        '''
        :param key: possible key values fct, ct30 and ct120
        '''
        res_path = os.path.join(self.paths['result_path'], 'SNR-Karte')
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if key == 'fct':
            name = 'fct_data.txt'
            header_string = f'T(proj.), SNR(Karte), d(Karte), theta(Berechnung(360/#proj.)), texp(Berechnung(snr_usr/snr_karte)), avg'
            merged_arr = hlp.merge_v1D(self.fCT_data['T'],
                                        self.fCT_data['snr'],
                                        self.fCT_data['d'],
                                        self.fCT_data['theta'],
                                        self.fCT_data['texp'],
                                        self.fCT_data['avg_num'])
        elif key == 'ct30':
            name = 'ct30_data.txt'
            header_string = f'T(proj.), SNR(proj.), d(Karte), theta(Berechnung(360/#proj.)), texp(Berechnung(snr_usr/snr_karte)), avg'
            merged_arr = hlp.merge_v1D(self.CT_data['T'],
                                       self.CT_data['snr'],
                                       self.CT_data['d'],
                                       self.CT_data['theta'],
                                       self.CT_data['texp'],
                                       self.CT_data['avg_num'])
        elif key == 'ct120':
            name = 'ct120_data.txt'
            header_string = f'T(proj.), SNR(proj.), d(Karte), theta(Berechnung(360/#proj.)), texp(base_texp), avg(no avg)'
            merged_arr = hlp.merge_v1D(self.CT_data['T'],
                                       self.CT_data['snr'],
                                       self.CT_data['d'],
                                       self.CT_data['theta'],
                                       self.CT_data['texp'],
                                       self.CT_data['avg_num'])
        np.savetxt(os.path.join(res_path, name), merged_arr, header=header_string)


    def reset_ct_data(self):
        self.CT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}


    def interpolate_avg_num(self, angles: int):
        CT_avg = []
        CT_texp =  []
        fCT_avg = self.fCT_data['avg_num']
        fCT_theta = self.fCT_data['theta']
        step = 360 / angles
        CT_theta = np.arange(0, 360, step)
        istep = int(CT_theta.size / fCT_theta.size)

        for i in range(fCT_avg.size):
            for j in range(istep):
                CT_avg.append(fCT_avg[i])
                CT_texp.append(fCT_avg[i] * self.BASE_texp)
        CT_avg = np.asarray(CT_avg)
        CT_texp = np.asarray(CT_texp)
        return CT_avg, CT_texp, CT_theta


    def calc_texp(self, snr: np.ndarray):
        t_exp = []
        avgs = []
        snr_multiplier = self.USR_SNR / snr

        for i in range(snr_multiplier.shape[0]):
            multiplier = round(self.USR_SNR / snr[i])
            if multiplier < self.AVG_MIN:
                multiplier = 1
            elif multiplier > self.AVG_MAX:
                multiplier = 4
            avgs.append(multiplier)
            t_exp.append(round(multiplier * self.BASE_texp, 2))
        return np.asarray(t_exp), np.asarray(avgs)


    def perform_mergings(self, key, img_stack):
        if key != 'darks' and key != 'refs':
            print('Only possble key values are \'darks\' or \'refs\'')
        base_path = os.path.dirname(self.paths['CT_imgs'])
        fin_dir = os.path.join(base_path, 'merged-CT', key)
        if not os.path.exists(fin_dir):
            os.makedirs(fin_dir)

        i = 1
        while i <= self.AVG_MAX:
            start = 0
            j = 0
            while j < img_stack.shape[0]:
                subdir = f'avg_{i}'
                end = start + i
                avg_img = hlp.calculate_avg(img_stack[start:end])
                path_and_name = os.path.join(fin_dir, subdir, f'{key}-avg{i}-{start}-{end-1}.raw')
                file.image.save(image=avg_img, filename=path_and_name, suffix='raw', output_dtype=np.uint16)
                start += i
                j += i
                if end + i > img_stack.shape[0]:
                    break
            i += 1
            hlp.rm_files(path=os.path.join(fin_dir, subdir), extension='info')
        self.MAX_CORR = len(next(os.walk(os.path.join(fin_dir, subdir)))[2])
        self.paths[f'CT_avg_{key}'] = fin_dir



    def activate_map(self):
        self.map = self.generator(spatial_range=self._ssize)
        self.map['T_min'] = self.T_min
        self.map['iU0'] = {'x': None, 'y': None, 'd': None}

        self.create_U0_curve(U0=self._U0)

        self.map['iU0']['x'], \
        self.map['iU0']['y'], \
        self.map['iU0']['d'] = self.find_intercept(kv_val=self._U0, T_val=self.T_min)


    def printer(self):
        kV_opt = self.map['Ubest_curve']['Ubest_val']
        print('\noptimal voltage for measurement:\n' + f'{kV_opt} kV' + '\n')


    def plot_map_overview(self):
        import matplotlib
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot()

        for d in self.map['d_curves']:
            _c_fit = self.map['d_curves'][d]['full']

            if hlp.is_int(d) and d in self.map['ds']:
                _c_data = self.map['d_curves'][d]['raw_data']
                _a = 1.0
                ax.plot(_c_fit[:, 1], _c_fit[:, 3], linestyle='-', alpha=_a, label=f'{d} mm')
                ax.scatter(_c_data[:, 1], _c_data[:, 2], marker='o', alpha=_a)
            else:
                _c = '#BBBBBB'
                _a = 0.15
                ax.plot(_c_fit[:, 1], _c_fit[:, 3], linestyle='-', linewidth=0.8, alpha=_a, c=_c)

        ax.axvline(x=self.map['T_min'], color='k', linestyle='--', linewidth=1)
        _c_U0 = self.map['U0_curve']['raw_data']
        _c_Ubest = self.map['Ubest_curve']['raw_data']
        ax.plot(_c_U0[:, 0], _c_U0[:, 1], linewidth=1.5, label=r'$U_{0}$')
        ax.plot(_c_Ubest[:, 0], _c_Ubest[:, 1], linewidth=1.5, label=r'$U_{\text{opt}}$')
        ax.legend(loc="upper left")
        ax.set_yscale('log')
        ax.set_xlabel('Transmission [w.E.]')
        ax.set_ylabel(r'SNR $[\text{s}^{-1}]$')
        fig.show()


    def check_practicability(self, avg_arr):
        threshold = 4
        for i in range(avg_arr.size):
            if threshold < avg_arr[i]:
                print(f'threshold value is set to {threshold}. But values in avg_array seem to extend it.')
                print(f'Change usr_snr or spatial range for the evaluation.')
                break