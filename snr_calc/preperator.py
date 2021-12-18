import csv
import datetime
import os
import gc
import helpers as hlp
import numpy as np
import matplotlib.pyplot as plt
from ext import file
from ext.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer

DETAILED = True


class SNRPrepperator:
    def __init__(self, path: str, path_result: str, magnification: float, image_holder, pixel_size_units: str = None,
                 watt: float = None, detector_pixel: float = None, smallest_size: float = None, nbins: int = None,
                 ex_kvs: list = None, ex_ds: list = None, only_snr: bool = False, snr_result_name: str = None,
                 trans_result_name: str = None):
        """
        see __init__() for the individual types of expected inputs of parameters
        :param img_shape: (detector height, detector width) in pixels
        :param header: image header size in bytes.
        :param path: a string to the path where the raw data (images) from measurement is located
        :param crop_area: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes for the crop area
                for the desired area you want to evaluate
        :param watt: adjusted watts for the measurement
        :param t_exp: adjusted exposure time for the measurement in ms
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
        self.path_base = path
        self.path_result = path_result
        self.img_holder = image_holder
        self.M = magnification

        if detector_pixel is not None:
            self.det_pixel = detector_pixel
        else:
            self.det_pixel = 74.8

        self.pixel_size = self.det_pixel / self.M
        if pixel_size_units is not None:
            self.pixel_size_units = pixel_size_units
        else:
            self.pixel_size_units = '$\mu m$'

        if nbins is not None:
            self.nbins = nbins
        else:
            self.nbins = 'auto'

        if watt is not None:
            self.watt = watt
        else:
            self.watt = 'unknown'

        self.x_min = smallest_size

        self.ex_kvs = []
        if ex_kvs is not None:
            self.ex_kvs = ex_kvs

        self.ex_ds = []
        if ex_ds is not None:
            self.ex_ds = ex_ds

        self.only_snr = only_snr
        if self.only_snr:
            print('only_snr = True --> No Transmission calc.')

        self.c_date = datetime.datetime.now()
        if snr_result_name is not None:
            self.SNR_name = snr_result_name
        self.SNR_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_SNR'
        if trans_result_name is not None:
            self.T_name = trans_result_name
        self.T_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_T'

    def __call__(self):
        self.calc_2D_snr(properties=self.get_properties())


    def calc_2D_snr(self, properties):
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
                    T = calc_T(imgs_data, imgs_refs, imgs_darks)
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


    def evaluate_snr(self, snr_obj, path_save_SNR, fig, data, refs, darks, _d, kV, t_exp, filterer):

        snr_obj.estimate_SNR(data, refs, darks, exposure_time=t_exp,
                             pixelsize=self.pixel_size, pixelsize_units=self.pixel_size_units,
                             series_filterer=filterer, u_nbins=self.nbins,
                             save_path=os.path.join(path_save_SNR,
                                                    f'SNR(x)_{self.watt}W_{kV}kV_{_d}mm_texp{t_exp}s'))

        return snr_obj, fig


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

    def constrains(self):
        pass


def calc_T(data, refs, darks):
    darks_avg = np.nanmean(darks, axis=0)
    refs_avg = np.nanmean(refs, axis=0)
    data_avg = np.nanmean(data, axis=0)
    img = (data_avg - darks_avg) / (refs_avg - darks_avg)

    transmission_min = 0
    h = 20
    w = 20
    median = []
    for i in range(0, img.shape[0]-h, h):
        for j in range(0, img.shape[1]-w, w):
            rect = img[i:i+h, j:j+w]
            medn = np.median(rect)
            median.append(medn)
    transmission_min = min(median)
    del img
    gc.collect()
    return transmission_min




class ImageLoader:
    def __init__(self, used_SCAP: bool = True, remove_lines: bool = True, load_px_map: bool = False,
                 crop_area: tuple = None):
        """
        ATTENTION:
        please note ref images MUST be loaded as first image stack! Since the ratio between median intensity of the
        stack and the outlier pixel rows is most significant at ref images.

        :param used_SCAP: set value to True if you captured your images with the x-ray source in-house software SCAP.
        This is important, since captured images with 'Metric_Steuerung' Software are flipped and rotated in compare to
        SCAP images.
        :param remove_lines: if is True, detector slice line will be removed.
        """
        self.used_SCAP = used_SCAP
        self.remove_lines = remove_lines

        self.header = 2048
        self.shape = (1536, 1944)
        if self.used_SCAP:
            self.header = 0
            self.shape = (1944, 1536)
        self.images = None

        if crop_area is not None:
            self.crop_area = crop_area
            self.view = (0, 0), *self.crop_area
        else:
            self.crop_area = (None, None)
            self.view = (None, None, None)

        self.idxs = []

        self.bad_px_map = load_px_map

        self.t_exp = None

        self.modified_px_map = None
        self.new_img_shape = None

        if self.bad_px_map:
            self.px_map = hlp.load_bad_pixel_map(crop=self.crop_area)
            self.filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.px_map)
        else:
            self.filterer = ImageSeriesPixelArtifactFilterer()


    def load_stack(self, path, stack_range = None):
        if not stack_range:
            self.images = file.volume.Reader(path,
                                             mode='raw',
                                             shape=self.shape,
                                             header=self.header,
                                             crops=self.view,
                                             dtype='<u2').load_all()
        else:
            self.images = file.volume.Reader(path,
                                             mode='raw',
                                             shape=self.shape,
                                             header=self.header,
                                             crops=self.view,
                                             dtype='<u2').load_range((stack_range[0], stack_range[-1]))
        if self.remove_lines:
            self.images = self.remove_detector_lines(self.images)
        return self.images


    def remove_detector_lines(self, img_stack):
        if len(self.idxs) < 1:
            DEVIATION_THRESHOLD = 0.15

            probe_img = img_stack[0]

            if self.used_SCAP:
                start, end = 0, probe_img.shape[1]
            else:
                start, end = 0, probe_img.shape[0]

            line_pos = 100
            line_plot = probe_img[line_pos-5:line_pos, start:end]
            line_plot = np.nanmean(line_plot, axis=0)
            line_median = np.nanmedian(line_plot)

            for i in range(len(line_plot)):
                px_val = line_plot[i]
                ratio = abs(1 - (px_val/line_median))

                if DEVIATION_THRESHOLD < ratio:
                    self.idxs.append(i)

        img_stack = np.delete(img_stack, self.idxs, axis=2)
        if self.bad_px_map:
            if self.px_map.shape[1] != img_stack.shape[2]: # only if the px_map was not updated yet, crop it to the new size
                self.px_map = np.delete(self.px_map, self.idxs, axis=1)
                self.filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.px_map) # updating the filterer with new map
        self.new_img_shape = img_stack.shape
        return img_stack


    def load_filterer(self):
        if self.bad_px_map is True:
            filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=self.modified_px_map)
            return filterer
        else:
            filterer = ImageSeriesPixelArtifactFilterer()
            return filterer


    def get_t_exp(self):
        for file in os.listdir(self.path):
            piece_l = file.split('expTime_')[1]
            piece_r = piece_l.split('__')[0]
            t_exp = int(piece_r)
            self.t_exp = t_exp / 1000
            break

    def get_shape(self):
        print((len(self.images), self.shape[0], self.shape[1]))


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
