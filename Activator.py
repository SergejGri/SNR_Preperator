import copy
import os
import sys

import numpy as np
from scipy import interpolate

from ct_operations import CT
from ct_operations import CT_multi_img
from snr_evaluator import SNREvaluator
import helpers as hlp
from Plotter import Plotter as PLT


class Activator:
    def __init__(self, image_loader, map_generator, attributes):
        self.paths = attributes.paths
        self.p_fin = os.path.join(os.path.dirname(self.paths['T_data']), 'MAP')
        self.fCT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}
        self.CT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}

        self._CT_steps = attributes.CT_steps
        self.imgs_per_angle = attributes.imgs_per_angle
        self.T_min = None
        self.mode_avg = False
        if attributes.base_texp is not None:
            self.mode_avg = True
            self._base_texp = attributes.base_texp

        self.sub_bin = attributes.snr_bins

        self.kv_ex = attributes.excluded_kvs
        self.ds_ex = attributes.excluded_thicknesses



        if attributes.spatial_size:
            self._ssize = attributes.spatial_size
        else:
            self._ssize = (100)
            self.init_MAP = True
            print('No spatial_size value was passed: Initial MAP creation @ 100E-6 m')
        self._snr_user = attributes.snr_user
        #self.scanner = scanner
        self.loader = image_loader
        #self.scanner = Scanner(params=attributes)

        self.stop_exe = False

        if 40 <= attributes.U0 <= 180:
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

        self.generator = map_generator
        #self.Generator = SNRMapGenerator(scanner=self.scanner, kv_filter=self.kv_ex)
        self._plt = PLT()


    def __call__(self, create_plot: bool = True, only_fCT: bool = True, detailed: bool = False):
        self.evaluate_fCT(image_loader=self.loader)
        self.activate_map()

        if self.map['iU0']['y'] is not None:
            self.create_Ubest_curve(d=self.map['iU0']['d'])
            self.printer()

        if not only_fCT:
            self.get_texp_from_fCT()
            self.evaluate_CT()

        if create_plot:
            p_fin = self.paths
            self._plt.T_kv_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)
            self._plt.map_plot(path_result=self.scanner.path_fin, object=self.map, detailed=detailed)
            self._plt.compare_fCT_CT(self)


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

        :param c1:  expects an array as input.

        :param c2:  expects an array or a value. This allows more flexibility in searching for intercepts
                    between curves and curves or between curves and const. values.

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
        self.map['U0_curve'] = {}
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
            avg_nmum = hlp.round_to_nearest_hundred(btexp, fCT_texp[i])
            loc_avgs.append(avg_nmum)

        return np.asarray(loc_avgs)


    def get_texp_from_fCT(self):
        fCT_T = self.fCT_data['T']

        self.fCT_data['T'], self.fCT_data['snr'], self.fCT_data['d'] = self.extract_MAP_data(kv_val=self._U0, T=fCT_T)

        self.fCT_data['texp'] = self.calc_texp(snr=self.fCT_data['snr'])
        self.fCT_data['avg_num'] = self.get_avgs_from_fCT(self._base_texp)


    def extract_MAP_data(self, kv_val: float, T: np.ndarray):
        '''
        searches at a given angle for snr, transmission and thickness
        :param kv_val:      voltage (curve) value at which you desire to find the intercept
        :param T:           transmission value at which you desire to find the intercept
        '''
        list_iT, list_isnr, list_id = [], [], []

        for i in range(T.size):
            iT, isnr, id = self.find_intercept(kv_val=kv_val, T_val=T[i])
            print(f'iT: {iT} \nisnr: {isnr} \nid: {id}')
            list_iT.append(iT), list_isnr.append(isnr), list_id.append(id)

        return np.asarray(list_iT), np.asarray(list_isnr), np.asarray(list_id)


    def evaluate_fCT(self, image_loader):
        imgs = image_loader.load_stack(path=self.paths['fCT_imgs'])
        refs = image_loader.load_stack(path=self.paths['fCT_refs'])
        darks = image_loader.load_stack(path=self.paths['fCT_darks'])

        self.fCT_data['T'] = CT(imgs=imgs, refs=refs, darks=darks, detailed=True)
        self.fCT_data['theta'] = hlp.gimme_theta(path_ct=self.paths['fCT_imgs'])
        self.T_min, _ = hlp.find_min(self.fCT_data['T'])


    def evaluate_CT(self):
        mode_dev = True
        loader_nh = copy.deepcopy(self.loader)
        loader_nh.header = 0  # must be assigned additionally, since merged imgs do not contain a header


        if self.mode_avg:
            list_images = [f for f in os.listdir(self.paths['CT_imgs']) if os.path.isfile(os.path.join(self.paths['CT_imgs'], f))]
            if self._CT_steps < len(list_images):
                self.CT_data['avg_num'], self.CT_data['theta'] = self.interpolate_avg_num(angles=50)
                if not mode_dev:
                    self.paths['CT_avg'] = CT_multi_img(self, self.paths['CT_imgs'], list_images, self.imgs_per_angle, img_loader=self.loader)

        # must be removed after dev
        self.paths['CT_avg'] = r'\\132.187.193.8\\junk\\sgrischagin\\2021-12-22-sergej-CT-halbesPhantom-102kV-100ms-5W-M4p46\\merged-CT'
        snr_evaluator = SNREvaluator(image_loader=loader_nh, watt=5.0, voltage=101, magnification=4.45914, btexp=self._base_texp, only_snr=False)

        # here projections need to be separated
        # -> not enough RAM for 1500 projections of shape(1500, 1944, 1536) and int64
        list_images = [f for f in os.listdir(self.paths['CT_avg']) if os.path.isfile(os.path.join(self.paths['CT_avg'], f))]
        view = slice(None, None), slice(50, 1500), slice(615, 1289)

        refs = self.loader.load_stack(path=self.paths['CT_refs'])
        darks = self.loader.load_stack(path=self.paths['CT_darks'])


        T_mins = []
        snrs = []
        j = 0
        angles = np.arange(0, 360, 360 / 50)
        print(angles)
        for i in range(0, len(list_images), 30):
            result_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_MA\3D_SNR_eval'
            # ATTENTION: projections must be loaded with image_loader where header is set to 0!
            imgs = loader_nh.load_stack(path=self.paths['CT_avg'], stack_range=(i, i+30))

            T_min, snr = snr_evaluator.snr_3D(result_path=result_path,
                                 data=imgs[view], refs=refs[view], darks=darks[view],
                                 exposure_time=self.CT_data['avg_num'][j],
                                 angle=angles[j])
            T_mins.append(T_min)
            snrs.append(snr)
            j += 1

        self.CT_data['T'] = np.asarray(T_mins)
        self.CT_data['snr'] = np.asarray(snrs)
        #self.CT_data['theta'] = hlp.gimme_theta(path_ct=self.paths['CT_avg'])

        Ubest = self.map['Ubest_curve']['Ubest_val']
        T = self.CT_data['T']
        self.CT_data['T'], self.CT_data['snr'], self.CT_data['d'] = self.extract_MAP_data(kv_val=Ubest, T=T)

        snr = self.CT_data['snr']
        self.CT_data['texp'] = self.calc_texp(snr=snr)
        self.CT_data['avg_num'], self.CT_data['theta'] = self.interpolate_avg_num(angles=50)


    def interpolate_avg_num(self, angles: int):
        fCT_avg = self.fCT_data['avg_num']
        fCT_theta = self.fCT_data['theta']

        step = 360 / angles
        CT_theta = np.arange(0, 360, step)

        CT_avg = []
        istep = int(CT_theta.size / fCT_theta.size)

        for i in range(fCT_avg.size):
            for j in range(istep):
                CT_avg.append(fCT_avg[i])

        CT_avg = np.asarray(CT_avg)
        return CT_avg, CT_theta


    def calc_texp(self, snr: np.ndarray):
        t_exp = []
        for i in range(snr.shape[0]):
            tmp_t = self._snr_user / snr[i]
            t_exp.append(tmp_t)
        return np.asarray(t_exp)


    def activate_map(self):
        self.map = self.generator(spatial_range=self._ssize)
        self.map['T_min'] = self.T_min
        self.map['iU0'] = {'x': None, 'y': None, 'd': None}

        self.create_U0_curve(U0=self._U0)

        self.map['iU0']['x'], \
        self.map['iU0']['y'], \
        self.map['iU0']['d'] = self.find_intercept(kv_val=self._U0, T_val=self.T_min)


    def printer(self):
        ix = self.map['iU0']['x']
        iy = self.map['iU0']['y']
        d_opt = self.map['iU0']['d']
        kV_opt = self.map['Ubest_curve']['Ubest_val']
        print(f'\noptimal voltage for measurement:\n' + f'{kV_opt} kV' + '\n')


    def smooth_curve(self, arr_1, arr_2):
        pass