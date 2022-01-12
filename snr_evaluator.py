import csv
import datetime
import os
import gc
import helpers as hlp
import numpy as np
from map_generator import SNRMapGenerator
from scanner import Scanner
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer

DETAILED = True


class SNREvaluator:
    def __init__(self, image_loader, watt: float, magnification: float, voltage: int, btexp: float, only_snr: bool,
                 pixel_size_units: str = None, detector_pixel: float = None, snr_result_name: str = None,
                 T_result_name: str = None):
        self.loader = image_loader
        self.M = magnification
        self.watt = watt
        self.voltage = voltage
        self.btexp = btexp
        self.EVA = SNR_Evaluator()
        self.filterer = ImageSeriesPixelArtifactFilterer()
        #self.scanner = Scanner()
        self.only_snr = only_snr

        if detector_pixel is not None:
            self.det_pixel = detector_pixel
        else:
            self.det_pixel = 74.8

        self.pixel_size = self.det_pixel / self.M
        if pixel_size_units is not None:
            self.pixel_size_units = pixel_size_units
        else:
            self.pixel_size_units = '$\mu m$'

        self.c_date = datetime.datetime.now()
        if snr_result_name is not None:
            self.SNR_name = snr_result_name
        else:
            self.SNR_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_SNR'
        if T_result_name is not None:
            self.T_name = T_result_name
        else:
            self.T_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_T'


    def calc_snr(self, images: np.ndarray, refs: np.ndarray, darks: np.ndarray, texp, px_size, px_units, filterer):
        self.EVA.estimate_SNR(images, refs, darks, exposure_time=self.btexp, pixelsize=self.pixel_size,
                              pixelsize_units=self.pixel_size_units, series_filterer=self.filterer,
                              save_path=os.path.join(self.SNR_name, f'SNR(x)_{self.watt}W_{self.voltage}kV_texp{self.btexp}s'))
        return self.EVA


    def calc_T(self, data, refs, darks):
        darks_avg = np.nanmean(darks, axis=0)
        refs_avg = np.nanmean(refs, axis=0)
        data_avg = np.nanmean(data, axis=0)
        img = (data_avg - darks_avg) / (refs_avg - darks_avg)
        h = 20
        w = 20
        median = []
        for i in range(0, img.shape[0] - h, h):
            for j in range(0, img.shape[1] - w, w):
                rect = img[i:i + h, j:j + w]
                medn = np.median(rect)
                median.append(medn)
        transmission_min = min(median)
        del img
        gc.collect()
        return transmission_min


    def snr_3D(self, result_path, data: np.ndarray, refs: np.ndarray, darks: np.ndarray, exposure_time, angle):
        res_str = f'SNR-{self.watt}W-{self.voltage}kV-@{angle}angle'
        results = []
        figure = None
        EVA = self.EVA
        texp = round(exposure_time * self.btexp, 3)


        T_min = self.calc_T(data, refs, darks)
        file = f'angle-{angle}.csv'
        _path_T = os.path.join(result_path, file)
        with open(os.path.join(_path_T), 'a+') as f:
            f.write('{};{}\n'.format(angle, T_min))
            f.close()

        sv_str = os.path.join(result_path, res_str)
        EVA.estimate_SNR(data, refs, darks, pixelsize=74.8, pixelsize_units='$\mu m$', exposure_time=texp,
                              series_filterer=self.filterer, save_path=os.path.join(result_path, res_str))
        figure = EVA.plot(figure, f'{angle}')
        results.append(EVA)
        EVA.finalize_figure(figure, save_path=os.path.join(result_path, f'angle-{angle}.pdf'))


        #agl = hlp.extract(what='angl', dfile=f)
        kv, snr = self.calc_avg_SNR(file=f, lb=lb, rb=rb)

        return T_min, snr



    def get_SNR_data(self, path_f, d, lb, rb):
        kvs = []
        snr_means = []

        list_files = [f for f in os.listdir(path_f) if os.path.isfile(os.path.join(path_f, f))]

        for file in list_files:
            _d = hlp.extract(what='angl', dfile=file)
            if _d == d:
                kV, mean_SNR = self.calc_avg_SNR(file, lb, rb)
                if kV in self.kV_filter:
                    del kV, mean_SNR
                else:
                    kvs.append(kV)
                    snr_means.append(mean_SNR)

        kv_arr = np.asarray(kvs).T
        snr_arr = np.asarray(snr_means).T
        arr = np.vstack((kv_arr, snr_arr)).T
        arr = arr[arr[:, 0].argsort()]
        return arr[:, 1]




    def write_T_data(self, path_T, d, T, voltage):
        file_l = f'{d}_mm.csv'
        if DETAILED:
            print(f'WRITING FILES {d} mm')

        _path_T = os.path.join(path_T, file_l)
        with open(os.path.join(_path_T), 'a+') as f_l:
            f_l.write('{};{}\n'.format(voltage, T))
            f_l.close()




class StepWedgeEvaluator(SNREvaluator):

    def __init__(self, path: str, path_result: str, ex_kvs: list = None, ex_ds: list = None, only_snr: bool = False):
        """
        see __init__() for the individual types of expected inputs of parameters
        :param img_shape: (detector height, detector width) in pixels
        :param header: image header size in bytes.
        :param path: a string to the path where the raw data (images) from measurement is located
        :param crop_area: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes for the crop area
                for the desired area you want to evaluate
        :param watt: adjusted watts for the measurement
        :param btexp: adjusted exposure time for the measurement in ms
        :param magnification: adjusted magnification for the measurement
        :param detector_pixel: pixel size of the detector. If no value is passed, a default value of 74.8 (current value
                of the MetRIC detector) will be used.
        :param pixel_size_units: a string which should be passed in latex form: $\mu m$ for micrometers etc. If no value
                is passed, default value of micrometer will be used.
        :param smallest_size: if smallest size is passed, the maximal value of the u-axis will be set to the passed
                value
        :param ex_kvs: expects a list of kV-folders which the user want to avoid. IF no list ist passed, the class will
            evaluate all voltage folders in the base_path
        :param mode_snr: default value True. If set to False, only transmission (as long as mode_T==True) will be
                evaluated
        :param mode_trans: default value True. Same behavior as mode_SNR
        :param snr_result_name: your desired name for SNR results
        :param trans_result_name: your desired name for Transmission results
        """
        super(StepWedgeEvaluator, self).__init__()
        self.path_base = path
        self.path_result = path_result

        self.ex_kvs = []
        if ex_kvs is not None:
            self.ex_kvs = ex_kvs

        self.ex_ds = []
        if ex_ds is not None:
            self.ex_ds = ex_ds

        self.only_snr = only_snr
        if self.only_snr:
            print('only_snr = True --> No Transmission calc.')




    def snr_2D_stepwedge(self):
        properties = self.get_properties()

        for dir in properties:
            voltage = hlp.extract(what='kv', dfile=dir)

            SNR_eval = SNR_Evaluator()

            path_darks = os.path.join(self.path_base, dir, 'darks')
            path_refs = os.path.join(self.path_base, dir, 'refs')
            imgs_refs = self.img_holder.load_stack(path=path_refs)
            imgs_darks = self.img_holder.load_stack(path=path_darks)

            psave_SNR = os.path.join(self.path_result, self.SNR_name, dir)
            psave_T = os.path.join(self.path_result, self.T_name)
            if not os.path.exists(psave_T):
                os.makedirs(psave_T)

            subdirs = properties[dir]['ds']
            t_exp = properties[dir]['t_exp']
            properties[dir]['path_snr'] = os.path.join(self.path_result, self.SNR_name, dir)
            properties[dir]['path_T'] = os.path.join(self.path_result, self.T_name)

            results = []
            figure = None

            for subdir in subdirs:
                _d = int(subdir)

                print(f'working on {dir}: {_d} mm')

                p_imgs, _, _ = self.prepare_imgs(self.path_base, dir, subdir)
                imgs_data = self.img_holder.load_stack(path=p_imgs)

                if not self.only_snr:
                    T = self.calc_T(imgs_data, imgs_refs, imgs_darks)
                    self.write_T_data(psave_T, _d, T, voltage)

                SNR_eval, figure = self.evaluate_snr(snr_obj=SNR_eval, path_save_SNR=psave_SNR, fig=figure,
                                                     data=imgs_data,
                                                     refs=imgs_refs,
                                                     darks=imgs_darks,
                                                     _d=_d, kV=voltage, t_exp=t_exp,
                                                     filterer=self.img_holder.filterer)
                figure = SNR_eval.plot(figure, f'{_d} mm')
                results.append(SNR_eval)

                if DETAILED:
                    print(f'Done with {dir}: {_d} mm')

            print('finalizing figure...')
            SNR_eval.finalize_figure(figure, title=f'{dir} @{self.watt}W',
                                     save_path=os.path.join(psave_SNR, f'{voltage}kV'))


    def get_properties(self):
        directories = {}
        for dirr in os.listdir(self.path_base):
            if os.path.isdir(os.path.join(self.path_base, dirr)) and 'kV' in dirr:
                kv = hlp.extract(what='kv', dfile=dirr)
                if kv in self.ex_kvs:
                    pass
                else:
                    subdirs = self.get_subdirs(dir=dirr)
                    t_exp = get_texp(path=self.path_base, kv=kv)
                    dict_sdir = {'ds': subdirs, 't_exp': t_exp}
                    directories[dirr] = dict_sdir
        return directories


    def get_subdirs(self, dir):
        subdirs = []
        working_dir = os.path.join(self.path_base, dir)

        for sdir in os.listdir(working_dir):
            if sdir.isdigit():
                if self.ex_ds is not None:
                    if int(sdir) in self.ex_ds:
                        pass
                    else:
                        subdirs.append(sdir)
        try:
            subdirs = [int(x) for x in subdirs]
            subdirs.sort()
            return subdirs
        except:
            print(f'Not correct naming? It seems that in your subdirs folder of {dir} have non numeric values \n-> '
                  f'sorting not possible.\n'
                  f'Please make sure your thickness folders are following the naming convention.\n')
            return 0


    def write_T_data(self, path_T, d, T, voltage):
        file_l = f'{d}_mm.csv'
        if DETAILED:
            print(f'WRITING FILES {d} mm')

        _path_T = os.path.join(path_T, file_l)
        with open(os.path.join(_path_T), 'a+') as f_l:
            f_l.write('{};{}\n'.format(voltage, T))
            f_l.close()


    @staticmethod
    def prepare_imgs(path, dir, subf):
        imgs = None
        ref_imgs = None
        dark_imgs = None
        if os.path.isdir(os.path.join(path, dir)):
            imgs = os.path.join(path, dir, str(subf))
            dark_imgs = os.path.join(path, dir, 'darks')
            ref_imgs = os.path.join(path, dir, 'refs')
        return imgs, ref_imgs, dark_imgs


    @staticmethod
    def filter_area_to_t(thick):
        return int(thick)



def get_t_exp_old(path):
    t_exp = None
    for file in os.listdir(path):
        piece_l = file.split('expTime_')[1]
        piece_r = piece_l.split('__')[0]
        t_exp = int(piece_r)
        t_exp = t_exp / 1000
        break
    return t_exp


def get_texp(path, kv):
    pfile = None
    for file in os.listdir(path):
        if file.endswith('.txt') and 'exp' in file:
            pfile = os.path.join(path, file)
    if isinstance(kv, str):
        kv = int(kv.split('kV')[0])

    if pfile:
        with open(pfile) as f:
            reader = csv.reader(f, delimiter=';')
            cols = list(zip(*reader))
            kv_col = cols[0]
            t_col = cols[1]
            idx = [i for i in range(len(kv_col)) if int(kv_col[i]) == kv][0]

            t_exp = int(t_col[idx])
    else:
        print('no file for exposure times were found. Pls make shure to put it in the same direction '
              'as the voltage folders and name it \'t_exp.txt\'')
    return t_exp / 1000


def get_voltage(dir):
    if '_' in dir:
        voltage = int(dir.split('_')[0])
    elif 'kV' in dir:
        voltage = int(dir.split('kV')[0])
    else:
        voltage = int(dir)
    return voltage





