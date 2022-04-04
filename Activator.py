import os
import sys
import numpy as np
from scipy import interpolate

from ct_operations import calc_T_for_stack
from ct_operations import merging_multi_img_CT
from snr_evaluator import SNREvaluator
from ext import file
from map_generator import SNRMapGenerator
from image_loader import ImageLoader
import helpers as hlp
from Plotter import Plotter as _plt
import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler







class Activator:

    DEBUG = True
    AVG_MAX = 16
    AVG_MIN = 1

    def __init__(self, attributes):
        self.paths = attributes.paths
        self.p_fin = attributes.paths['result_path']
        #self.p_fin = os.path.join(os.path.dirname(self.paths['MAP_T_files']), 'Karte')
        self.fCT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}
        self.CT_data = {'T': None, 'snr': None, 'd': None, 'theta': None, 'texp': None, 'avg_num': None}

        self.CT_steps = attributes.CT_steps
        self.fCT_steps = attributes.fCT_steps
        self.imgs_per_angle = attributes.imgs_per_angle
        self.realCT_steps = attributes.realCT_steps
        self.T_min = None
        self.mode_avg = False
        self.MAX_CORR_IMGS = 0
        if attributes.base_texp is not None:
            self.mode_avg = True
            self.BASE_texp = attributes.base_texp

        self.mergings = attributes.new_mergings
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

        self.view = slice(None, None), slice(133, 945), slice(672, 1220)

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

        if create_plot:
            self._plt.T_kv_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            self._plt.snr_kv_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            self._plt.map_plot(path_result=self.p_fin, object=self.map, detailed=detailed)
            plot_algo_stepwise(self.map)
        self.extract_data_for_fCT()
        self.reset_ct_data()
        self.evaluate_CT_merge()
        self.reset_ct_data()
        self.evaluate_CT_no_merge()
        #self.reset_ct_data()



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
        old_d = None
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
                old_d = d
            elif delta < old_delta:
                old_delta = delta
                _d = d
        if _d is None:
            _d = old_d
        print('test')
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
        self.write_avglist_for_kilian(N=1500)


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
        self.fCT_data['T'], self.fCT_data['theta'] = calc_T_for_stack(imgs=imgs, refs=refs, darks=darks, detailed=True)
        self.T_min, _ = hlp.find_min(self.fCT_data['T'])


    # 30er schritte
    def evaluate_CT_merge(self):

        loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
        loader_nh = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
        loader_nh.header = 0  # must be assigned additionally, since merged imgs do not contain a header

        self.CT_data['avg_num'], self.CT_data['texp'], self.CT_data['theta'] = self.interpolate_avg_num(angles=self.CT_steps)
        self.write_mergings()
        if self.mergings:
            print('\n\nperforming merging of images\n\n')
            list_images = [f for f in os.listdir(self.paths['CT_imgs']) if os.path.isfile(os.path.join(self.paths['CT_imgs'], f))]
            l_images = sorted(list_images, key=lambda x: int(x.partition('_none__')[2].partition('.raw')[0]))
            self.paths['CT_avg_imgs'] = merging_multi_img_CT(self, self.paths['CT_imgs'], l_images, self.imgs_per_angle, img_loader=loader)
            #self.paths['CT_avg_imgs'] = r'\\132.187.193.8\junk\sgrischagin\2022-02-26-sergej-AluFlakes\gated-CT-50kV-480proj-50angles_new_mergings\imgs'
            # self.perform_mergings(img_stack=refs, key='refs')
            # self.perform_mergings(img_stack=darks, key='darks')

        refs = loader.load_stack(path=self.paths['CT_refs'])
        darks = loader.load_stack(path=self.paths['CT_darks'])
        refs = refs[self.view]
        darks = darks[self.view]

        snr_evaluator = SNREvaluator(watt=5.0, voltage=50, magnification=4.05956)
        Ts = []
        snrs = []
        snrst = []
        angles = np.arange(0, 360, 360 / self.CT_steps)

        print('performing SNR evaluations on merged images')
        for i, j in zip(range(0, 24000, self.sub_bin), np.arange(0, self.CT_steps)):
            # ATTENTION: projections must be loaded with image_loader where header is set to 0!
            avg = self.CT_data['avg_num'][j]
            texp = round(self.CT_data['texp'][j], 2)
            theta = round(angles[j], 1)
            imgs = loader_nh.load_stack(path=self.paths['CT_avg_imgs'], stack_range=(i, i+self.sub_bin))

            imgs = imgs[self.view]
            transmission, snr = snr_evaluator.snr_3D(generator=self.generator,
                                              result_path=self.paths['result_path'] + r'\SNRm',
                                              data=imgs, refs=refs, darks=darks, texp=texp, angle=theta)
            snrt = snr * texp

            snrs.append(snr)
            snrst.append(snrt)
            Ts.append(transmission)

        self.CT_data['T'] = np.asarray(Ts)
        self.CT_data['snr'] = np.asarray(snrs)
        self.CT_data['snrt'] = np.asarray(snrst)

        self.write_data(key='avg')


    def evaluate_CT_no_merge(self):
        print('\n\nperforming SNR evaluations on non merged images\n\n')
        # 120 schritte
        Ts = []
        snrs = []
        snrst = []
        loader = ImageLoader(used_SCAP=False, remove_lines=False, load_px_map=False)
        snr_evaluator = SNREvaluator(watt=5.0, voltage=50, magnification=4.05956)
        #list_images = [f for f in os.listdir(self.paths['CT_imgs']) if os.path.isfile(os.path.join(self.paths['CT_imgs'], f))]
        list_images = 24000
        angles = np.arange(0, 360, 360 / self.CT_steps)

        refs = loader.load_stack(path=self.paths['CT_refs'])
        darks = loader.load_stack(path=self.paths['CT_darks'])
        refs = refs[self.view]
        darks = darks[self.view]

        STEP = 480
        if self.BASE_texp is not None:
            texp = self.BASE_texp
        else:
            texp = 0.05
        j = 0
        for i in range(0, list_images, STEP):      # after debugmode change to range(0, len(list_images), STEP):
            print(f'ct120: {round( j*STEP/list_images ,2)*100} done')
            imgs = loader.load_stack(path=self.paths['CT_imgs'], stack_range=(i, i + STEP))
            imgs = imgs[self.view]
            transmission, snr = snr_evaluator.snr_3D(generator=self.generator,
                                              result_path=self.paths['result_path'] + r'\SNRo',
                                              data=imgs, refs=refs, darks=darks, texp=texp,
                                              angle=round(angles[j], 1))
            snrt = snr * texp

            snrs.append(snr)
            snrst.append(snrt)
            Ts.append(transmission)
            j += 1
        self.CT_data['T'] = np.asarray(Ts)
        self.CT_data['snr'] = np.asarray(snrs)
        self.CT_data['snrt'] = np.asarray(snrst)
        self.write_data_small(key='data_no_merge')

        #Ubest = self.map['Ubest_curve']['Ubest_val']
        #_, _, self.CT_data['d'] = self.extract_MAP_data(kv_val=Ubest, T=self.CT_data['T'])
        _, _, self.CT_data['theta'] = self.interpolate_avg_num(angles=self.CT_steps)
        self.CT_data['texp'] = np.empty(self.CT_data['snr'].size)
        self.CT_data['texp'].fill(self.BASE_texp)
        self.CT_data['avg_num'] = np.empty(self.CT_data['snr'].size)
        self.CT_data['avg_num'].fill(1)
        self.write_data(key='not_avg')


    def write_data(self, key):
        '''
        :param key: possible key values fct, ct30 and ct120
        '''
        res_path = os.path.join(self.paths['result_path'], 'SNR-Karte')
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if key == 'fct':
            name = 'fct_data.txt'
            header_string = r'T(proj.), fCT_SNR(Karte), d(Karte), theta(Berechnung(360/#proj.)), texp(Berechnung(snr_usr/snr_karte)), avg'
            merged_arr = hlp.merge_v1D(self.fCT_data['T'],
                                        self.fCT_data['snr'],
                                        self.fCT_data['d'],
                                        self.fCT_data['theta'],
                                        self.fCT_data['texp'],
                                        self.fCT_data['avg_num'])
        elif key == 'avg':
            name = 'avg_snr.txt'
            header_string = r'T(proj.), SNRm(not multiplied by texp), SNRt, d(Karte), theta(Berechnung(360/#proj.)), texp(Berechnung(snr_usr/snr_karte)), avg'
            merged_arr = hlp.merge_v1D(self.CT_data['T'],
                                       self.CT_data['snr'],
                                       self.CT_data['snrt'],
                                       self.CT_data['theta'],
                                       self.CT_data['texp'],
                                       self.CT_data['avg_num'])
        elif key == 'not_avg':
            name = 'ct480_data.txt'
            header_string = r'T(proj.), SNRo(not multiplied by texp), SNRt, d(Karte), theta(Berechnung(360/#proj.)), texp(base_texp), avg(no avg)'
            merged_arr = hlp.merge_v1D(self.CT_data['T'],
                                       self.CT_data['snr'],
                                       self.CT_data['snrt'],
                                       self.CT_data['theta'],
                                       self.CT_data['texp'],
                                       self.CT_data['avg_num'])
        np.savetxt(os.path.join(res_path, name), merged_arr, header=header_string)


    def write_data_small(self, key):
        res_path = os.path.join(self.paths['result_path'], 'SNR-Karte')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        name = f'{key}_data_small.txt'
        header_string = f'T(proj.), SNR(proj.)'
        merged_arr = hlp.merge_v1D(self.CT_data['T'],
                                   self.CT_data['snr'])

        np.savetxt(os.path.join(res_path, name), merged_arr, header=header_string)



    def write_mergings(self):
        res_path = os.path.join(self.paths['result_path'], 'SNR-Karte')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        name = f'avgs.txt'
        header_string = f'avgs, texp, theta'
        merged_arr = hlp.merge_v1D(self.CT_data['avg_num'], self.CT_data['texp'], self.CT_data['theta'])

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


    def write_avglist_for_kilian(self, N):
        res_path = r'C:\Users\Sergej Grischagin\Desktop\final_evaluations\3D_SNR_eval_10032022_biggerview_Cylinder_v1_ss35\SNR-Karte'
        name = rf'avgs-for-CT-{N}proj.txt'
        header_string = r'# theta, texp, avg'
        avg, texp, theta = self.interpolate_avg_num(angles=N)
        merged_arr = hlp.merge_v1D(theta, texp, avg)
        np.savetxt(os.path.join(res_path, name), merged_arr, header=header_string)


    def calc_texp(self, snr: np.ndarray):
        t_exp = []
        avgs = []
        snr_multiplier = self.USR_SNR / snr

        for i in range(snr_multiplier.shape[0]):
            multiplier = round(self.USR_SNR / snr[i])
            if multiplier < self.AVG_MIN:
                multiplier = 1
            elif multiplier > self.AVG_MAX:
                multiplier = 16
            avgs.append(multiplier)
            t_exp.append(round(multiplier * self.BASE_texp, 2))
        return np.asarray(t_exp), np.asarray(avgs)


    def perform_mergings(self, key, img_stack):
        if key != 'darks' and key != 'refs':
            print('Only possble key values are \'darks\' or \'refs\'')
        base_path = os.path.dirname(self.paths['CT_avg_imgs'])
        fin_dir = os.path.join(base_path, key)
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
        self.MAX_CORR_IMGS = len(next(os.walk(os.path.join(fin_dir, subdir)))[2])
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


    def check_practicability(self, avg_arr):
        threshold = 4
        for i in range(avg_arr.size):
            if threshold < avg_arr[i]:
                print(f'threshold value is set to {threshold}. But values in avg_array seem to extend it.')
                print(f'Change usr_snr or spatial range for the evaluation.')
                break






def plot_map_ubest(obj):
    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]

    # ============================================= SCHRITT 1: RAW DATA ===========================================
    for d in obj['d_curves']:
        if d in raw_mm:
            x = obj['d_curves'][d]['raw_data'][:, 1]
            y = obj['d_curves'][d]['raw_data'][:, 2]
            plt.scatter(x, y, label=f'{d} mm')
    plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\overview_plots\map_schritt1.pdf',
                bbox_inches='tight', dpi=600)



    # ========================================== SCHRITT 2: INTERPOLATE DATA ========================================
    plt.clf()
    plt.gca().set_prop_cycle(None)
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]


        if d in raw_mm:
            x_sc = obj['d_curves'][d]['raw_data'][:, 1]
            y_sc = obj['d_curves'][d]['raw_data'][:, 2]
            plt.scatter(x_sc, y_sc, zorder=10, marker='o', label='_nolegend_')
            plt.plot(x, y, zorder=5, label=f'{d} mm')
        else:
            plt.plot(x, y, linestyle='-', zorder=0, linewidth=1, c='#9e9e9e', alpha=0.5)
    plt.yscale('log')
    plt.xlabel(r'Transmission [w. E.]')
    plt.ylabel(r'SNR [$s^{-1}$]')
    plt.legend(loc='lower right', fancybox=True, shadow=False, ncol=1)

    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\overview_plots\map_schritt2.pdf',
                bbox_inches='tight', dpi=600)



def plot_algo_stepwise(obj):
    pgf_with_latex = {  # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
        "text.usetex": True,  # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],  # blank entries should cause plots
        "font.sans-serif": [],  # to inherit fonts from the document
        "font.monospace": [],
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "font.size": 12,
        "legend.fontsize": 12,  # Make the legend/label fonts
        "xtick.labelsize": 12,  # a little smaller
        "xtick.bottom": False,
        "xtick.top": False,
        "ytick.right": False,
        "ytick.left": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "ytick.labelsize": 12,
        "pgf.preamble": "\n".join([r"\usepackage{libertine}",
                                   r"\usepackage[libertine]{newtxmath}",
                                   r"\usepackage{siunitx}",
                                   r"\usepackage[utf8]{inputenc}",
                                   r"\usepackage[T1]{fontenc}"])
    }

    mpl.use("pgf")
    mpl.rcParams.update(pgf_with_latex)
    plt.rcParams["figure.figsize"] = (6.3, 3.13)

    plt.rc('lines', linewidth=1)
    custom_cycler = cycler('linestyle', ['-', '--']) * cycler('color', ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97', '#474747', '#9e9e9e'])
    plt.rc('axes', prop_cycle=custom_cycler)

    ms = 5
    txt_x = 0.12
    txt_y = 5
    lw_small_ds = 0.1
    lw_big_ds = 1

    raw_mm = [0, 2, 4, 8, 12, 16, 20, 24, 28]
    fig, ax = plt.subplots()
    plt.gca().set_prop_cycle(None)

    # ============================================= SCHRITT 1: RAW DATA ===========================================
    for d in obj['d_curves']:
        if d in raw_mm:
            x = obj['d_curves'][d]['raw_data'][:, 1]
            y = obj['d_curves'][d]['raw_data'][:, 2]
            plt.scatter(x, y, s=ms)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.annotate(r'\textbf{Schritt 1}', xy=(txt_x, txt_y), zorder=15, bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5), ha='center', va='bottom')
    ax.annotate(r'\SI{50}{\kilo\volt}', xy=(0.5, 0.1), zorder=15,
                bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5), ha='center', va='bottom')
    ax.annotate(r'\SI{60}{\kilo\volt}', xy=(0.7, 0.1), zorder=15,
                bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5), ha='center', va='bottom')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt1.pdf',
                bbox_inches='tight', dpi=600)




    plt.gca().set_prop_cycle(None)
    plt.cla()
    # ========================================= SCHRITT 2: INTERPOLATION RAWS =======================================
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]
        if d in raw_mm:
            plt.plot(x, y, zorder=10)
            plt.scatter(obj['d_curves'][d]['raw_data'][:, 1], obj['d_curves'][d]['raw_data'][:, 2], marker='o', s=ms, zorder=10)
        #else:
        #    plt.plot(x, y, linewidth=lw_small_ds, linestyle='-', zorder=0, c='#BBBBBB')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.annotate(r'\textbf{Schritt 2}', xy=(txt_x, txt_y), zorder=15, bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5),
                ha='center', va='bottom')

    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt2.pdf',
                bbox_inches='tight', dpi=600)




    # ========================================= SCHRITT 3: U0-T_min intercept =======================================
    plt.gca().set_prop_cycle(None)
    plt.cla()
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]
        if d in raw_mm:
            plt.plot(x, y, zorder=10)
            plt.scatter(obj['d_curves'][d]['raw_data'][:, 1], obj['d_curves'][d]['raw_data'][:, 2], marker='o', s=ms, zorder=10)
        else:
            plt.plot(x, y, linestyle='-', linewidth=lw_small_ds, zorder=0, c='#BBBBBB')
    plt.plot(obj['U0_curve']['raw_data'][:, 0], obj['U0_curve']['raw_data'][:, 1], linestyle='--', linewidth=1, c='k', zorder=15, label='$U_{0}$')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.axvline(x=obj['iU0']['x'], color='k', linestyle='-', linewidth=0.5, zorder=15)
    ax.annotate(r'\textbf{Schritt 3}', xy=(txt_x, txt_y), zorder=15, bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5), ha='center', va='bottom')

    plt.yscale('log')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt3.pdf',
                bbox_inches='tight', dpi=600)




    # ========================================= SCHRITT 4: SELECTION CURVE for U0 =======================================
    plt.gca().set_prop_cycle(None)
    plt.cla()
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]
        if d in raw_mm:
            plt.plot(x, y, zorder=10)
            plt.scatter(obj['d_curves'][d]['raw_data'][:, 1], obj['d_curves'][d]['raw_data'][:, 2], marker='o', s=ms,
                        zorder=10)
        elif d == obj['iU0']['d']:
            plt.plot(x, y, linestyle='-', linewidth=0.75, zorder=10, c='#FF2C00')
            idx = np.argmax(y)
            plt.scatter(x[idx], y[idx], marker='+', s=5, c='#FF2C00')
        else:
            plt.plot(x, y, linestyle='-', linewidth=lw_small_ds, zorder=0, c='#BBBBBB')

    plt.plot(obj['U0_curve']['raw_data'][:, 0], obj['U0_curve']['raw_data'][:, 1], linestyle='--', linewidth=1, c='k',
             zorder=15, label='$U_{0}$')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.axvline(x=obj['iU0']['x'], color='k', linestyle='-', linewidth=0.5, zorder=15)
    ax.annotate(r'\textbf{Schritt 4}', xy=(txt_x, txt_y), zorder=15, bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5), ha='center', va='bottom')

    plt.yscale('log')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt4.pdf',
                bbox_inches='tight', dpi=600)




    # ========================================= SCHRITT 5: CURVE U0 =======================================
    plt.gca().set_prop_cycle(None)
    plt.cla()
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]
        if d in raw_mm:
            plt.plot(x, y, zorder=10)
            plt.scatter(obj['d_curves'][d]['raw_data'][:, 1], obj['d_curves'][d]['raw_data'][:, 2], marker='o', s=ms,
                        zorder=10)
        elif d == obj['iU0']['d']:
            plt.plot(x, y, linestyle='-', linewidth=0.75, zorder=10, c='#FF2C00')
            idx = np.argmax(y)
            plt.scatter(x[idx], y[idx], marker='+', s=5, c='#FF2C00')
        else:
            plt.plot(x, y, linestyle='-', linewidth=lw_small_ds, zorder=0, c='#BBBBBB')

    plt.plot(obj['U0_curve']['raw_data'][:, 0], obj['U0_curve']['raw_data'][:, 1], linestyle='--', linewidth=1, c='k', zorder=15, label='$U_{0}$')
    plt.plot(obj['Ubest_curve']['raw_data'][:, 0], obj['Ubest_curve']['raw_data'][:, 1], linestyle='-', linewidth=1, c='k', zorder=15, label=r'$U_{\text{opt}}$')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.annotate(r'\textbf{Schritt 5}', xy=(txt_x, txt_y), zorder=15, bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5),
                ha='center', va='bottom')

    plt.yscale('log')
    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt5.pdf',
                bbox_inches='tight', dpi=600)




    # ========================================= SCHRITT 6: SNR(T) WERTE AUSLESEN =======================================
    plt.gca().set_prop_cycle(None)
    plt.cla()
    for d in obj['d_curves']:
        x = obj['d_curves'][d]['full'][:, 1]
        y = obj['d_curves'][d]['full'][:, 3]
        if d in raw_mm:
            plt.plot(x, y, zorder=10)
            plt.scatter(obj['d_curves'][d]['raw_data'][:, 1], obj['d_curves'][d]['raw_data'][:, 2], marker='o', s=ms,
                        zorder=10)
        elif d == obj['iU0']['d']:
            plt.plot(x, y, linestyle='-', linewidth=0.75, zorder=10, c='#FF2C00')
            idx = np.argmax(y)
            plt.scatter(x[idx], y[idx], marker='+', s=5, c='#FF2C00')
        else:
            plt.plot(x, y, linestyle='-', linewidth=lw_small_ds, zorder=0, c='#BBBBBB')

    plt.plot(obj['Ubest_curve']['raw_data'][:, 0], obj['Ubest_curve']['raw_data'][:, 1], linestyle='-', linewidth=1, c='k', zorder=15, label=r'$U_{\text{opt}}$')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    x1, y1 = [0.114, 0.114], [0, 0.001]
    x2, y2 = [-0.05, -0.01], [0.238, 0.238]

    #plt.axhline(y=0.237, xmin=0.0, xmax=0.025, color='#FF2C00', linestyle='-', linewidth=0.5, zorder=15)
    #plt.axvline(x=0.114, ymin=0.0, ymax=0.02, color='#FF2C00', linestyle='-', linewidth=0.5, zorder=15)
    plt.yscale('log')

    ax.annotate(r'\textbf{Schritt 6}', xy=(txt_x, txt_y), zorder=15,
                bbox=dict(boxstyle="round", fc="w", ec="#BBBBBB", alpha=0.5),
                ha='center', va='bottom')

    ax.annotate(r'$SNR(T, \theta)$', xy=(-0.02, 0.238), zorder=15,
                ha='center', va='bottom', color='#FF2C00', rotation=90, fontsize=5)

    ax.annotate(r'$T(\theta)$', xy=(0.118, 0.018), zorder=15,
                ha='center', va='bottom', color='#FF2C00', fontsize=5)

    plt.legend(loc=4)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Sergej Grischagin\PycharmProjects\SNR_Preperator\Algo_steps\map_schritt6.pdf',
                bbox_inches='tight', dpi=600)



    print('test')