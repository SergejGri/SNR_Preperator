import csv
import datetime
import os
import gc
import timeit
import helpers as h
import numpy as np
from externe_files import file
from externe_files.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer


def check_px_vals(img):
    start = timeit.default_timer()
    for z in range(len(img)):
        if any(0.0 in x for x in img[z]):
            print(f'x=0.0 @ z:{z}')
        if any(1.0 in x for x in img[z]):
            print(f'x=1.0 @ z:{z}')
        else:
            pass
    stop = timeit.default_timer()
    print(f'Execution time: {round((stop - start), 2)}s')


class SNRPrepperator:
    def __init__(self, img_shape: tuple, header: int, path: str, path_result: str, magnification: float,
                 crop_area: tuple, stack_range: tuple = None, pixel_size_units: str = None, watt: float = None,
                 detector_pixel: float = None, smallest_size: float = None, nbins: int = None, exclude: list = None,
                 only_snr: bool = False, snr_result_name: str = None, trans_result_name: str = None):
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
        :param exclude: expects a list of kV-folders which the user want to avoid. IF no list ist passed, the class will
            evaluate all voltage folders in the base_path
        :param mode_snr: default value True. If set to False, only transmission (as long as mode_T==True) will be
                evaluated
        :param mode_trans: default value True. Same behavior as mode_SNR
        :param snr_result_name: your desired name for SNR results
        :param trans_result_name: your desired name for Transmission results
        """
        self.path_base = path
        self.path_result = path_result

        self.img_shape = img_shape
        if self.img_shape != (1536, 1944):
            print(f'Since your passed detector size of {self.img_shape} differ from my default expectation of '
                  f'(1536, 1944), i do not know which area you want to analyse => You need to pass values for slice_l '
                  f'and slice_r to get plausible results.')
        self.header = header
        self.crop_area = crop_area
        self.view = (0, 0), *self.crop_area

        self.img_params = self.get_img_params()

        self.stack_range = stack_range
        if self.stack_range is None:
            self.stack_range = (None, None)
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
        self.exclude = exclude

        self.only_snr = only_snr
        if self.only_snr:
            print('only_snr = True --> No Transmission calc.')

        self.c_date = datetime.datetime.now()
        if snr_result_name is not None: self.SNR_name = snr_result_name
        self.SNR_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_SNR'
        if trans_result_name is not None: self.T_name = trans_result_name
        self.T_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_T'

    def __call__(self):
        self.calc_2D_snr(properties=self.get_properties())


    def calc_2D_snr(self, properties):
        #px_map = h.load_bad_pixel_map(crop=self.crop_area)
        #filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=px_map)
        filterer = ImageSeriesPixelArtifactFilterer()

        for dir in properties:
            p_dark = os.path.join(self.path_base, dir, 'darks')
            p_ref = os.path.join(self.path_base, dir, 'refs')
            psave_SNR = os.path.join(self.path_result, self.SNR_name, dir)
            psave_T = os.path.join(self.path_result, self.T_name)
            if not os.path.exists(psave_T):
                os.makedirs(psave_T)

            subdirs = properties[dir]['ds']
            t_exp = properties[dir]['t_exp']
            properties[dir]['path_snr'] = os.path.join(self.path_result, self.SNR_name, dir)
            properties[dir]['path_T'] = os.path.join(self.path_result, self.T_name)

            SNR_eval = SNR_Evaluator()

            results = []
            figure = None

            darks = file.volume.Reader(p_dark, **self.img_params).load_all()
            refs = file.volume.Reader(p_ref, **self.img_params).load_all()
            for subdir in subdirs:
                _d = int(subdir)
                print(f'working on {dir}: {_d} mm')

                p_imgs, _, _ = self.prepare_imgs(self.path_base, dir, subdir)
                data = file.volume.Reader(p_imgs, **self.img_params).load_all()

                if not self.only_snr:
                    T = self.calc_T(data, refs, darks)
                    self.write_T_data(psave_T, _d, T, get_voltage(dir))

                SNR_eval, figure = self.evaluate_snr(snr_obj=SNR_eval, path_save_SNR=psave_SNR, fig=figure, data=data,
                                                     refs=refs, darks=darks, _d=_d, kV=get_voltage(dir), t_exp=t_exp,
                                                     filterer=filterer)
                figure = SNR_eval.plot(figure, f'{_d} mm')

                results.append(SNR_eval)
                del data
                gc.collect()
                print(f'Done with {dir}: {_d} mm')

            print('finalizing figure...')
            SNR_eval.finalize_figure(figure, title=f'{dir} @{self.watt}W',
                                     save_path=os.path.join(psave_SNR, f'{get_voltage(dir)}kV'))

    def get_properties(self):
        directories = {}
        dict_sdir = {}
        for dirr in os.listdir(self.path_base):
            if os.path.isdir(os.path.join(self.path_base, dirr)):
                kv = h.extract_kv(dirr)
                if os.path.isdir(os.path.join(self.path_base, dirr)) and 'kV' in dirr:
                    subdirs = self.get_subdirs(dir=dirr)
                    t_exp = get_texp(path=self.path_base, kv=kv)
                    dict_sdir = {'ds': subdirs, 't_exp': t_exp}
                directories[dirr] = dict_sdir
        return directories

    def get_subdirs(self, dir):
        subdirs = []
        working_dir = os.path.join(self.path_base, dir)

        for sdir in os.listdir(working_dir):
            if os.path.isdir(os.path.join(working_dir, sdir)):
                subdirs.append(sdir)
        if 'darks' in subdirs:
            subdirs.remove('darks')
        if 'refs' in subdirs:
            subdirs.remove('refs')

        subdirs = [int(x) for x in subdirs]
        subdirs.sort()
        return subdirs


    def get_img_params(self):
        params = dict(mode='raw', shape=self.img_shape, header=self.header, dtype='<u2', crops=self.view)
        return params

    def write_T_data(self, path_T, d, T, voltage):
        file_l = f'{d}_mm.csv'
        print(f'WRITING FILES {d} mm')

        _path_T = os.path.join(path_T, file_l)

        with open(os.path.join(_path_T), 'a+') as f_l:
            f_l.write('{};{}\n'.format(voltage, T))
            f_l.close()

    def calc_T(self, data, refs, darks):
        darks_avg = np.nanmean(darks, axis=0)
        refs_avg = np.nanmean(refs, axis=0)

        img = (data - darks_avg) / (refs_avg - darks_avg)
        T = img[np.where(img > 0)].mean()
        del img
        gc.collect()
        return T


    def evaluate_snr(self, snr_obj, path_save_SNR, fig, data, refs, darks, _d, kV, t_exp, filterer):
        snr_obj.estimate_SNR(data, refs, darks, exposure_time=t_exp,
                             pixelsize=self.pixel_size, pixelsize_units=self.pixel_size_units,
                             series_filterer=filterer, u_nbins=self.nbins,
                             save_path=os.path.join(path_save_SNR,
                                                    f'SNR(x)_{self.watt}W_{kV}kV_{_d}mm_texp{t_exp}s'))
        return snr_obj, fig

    def calc_3D_snr(self):
        # path_save_SNR = os.path.join(self.path_result, self.SNR_name, dir)
        # path_save_T = os.path.join(self.path_result, self.T_name)
        # if not os.path.exists(path_save_T):
        #    os.makedirs(path_save_T)

        refs = ImageHolder(
            path=r'\\132.187.193.8\junk\sgrischagin\2021-08-26-Sergej_SNR-Stufenkeil_130proj_6W\100kV\refs')

        SNR_eval = SNR_Evaluator()
        str_voltage = get_voltage(dir)

        results = []

        p_imgs = r''
        p_refs = r''
        p_darks = r''

        darks = file.volume.Reader(p_darks, mode='raw', shape=self.img_shape, header=self.header,
                                   dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))
        refs = file.volume.Reader(p_refs, mode='raw', shape=self.img_shape, header=self.header,
                                  dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))
        figure = None
        for a in subf:
            _d = self.filter_area_to_t(a)

            print(f'working on {dir}: {_d} mm')
            data = file.volume.Reader(p_imgs, mode='raw', shape=self.img_shape, header=self.header,
                                      dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))

            self.t_exp = get_t_exp(p_imgs)
            SNR_eval, figure = self.evaluate_snr(SNR_eval, p_imgs, path_save_SNR, figure, data, refs, darks, _d,
                                                 str_voltage)
            figure = SNR_eval.plot(figure, f'{_d} mm')
            results.append(SNR_eval)
            del data
            gc.collect()
            print(f'Done with {dir}: {_d} mm')

        print('finalizing figure...')
        SNR_eval.finalize_figure(figure, title=f'{dir}_@{self.watt}W', smallest_size=self.x_min,
                                 save_path=os.path.join(path_save_SNR, f'{str_voltage}kV'))








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



class ImageHolder:
    def __init__(self, path: np.array, range: tuple = None):
        self.path = path
        self.header = 2048
        self.shape = (1536, 1944)
        self.images = None
        self.stack_range = range
        self.t_exp = None

    def load_images(self):
        if not self.stack_range:
            self.images = file.volume.Reader(self.path, mode='raw', shape=self.shape, header=self.header,
                                             dtype='<u2').load_all()
        else:
            self.images = file.volume.Reader(self.path, mode='raw', shape=self.shape, header=self.header,
                                             dtype='<u2').load_range((self.stack_range[0], self.stack_range[-1]))
        return self.images




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
    return t_exp/1000


def get_voltage(dir):
    if '_' in dir:
        voltage = int(dir.split('_')[0])
    elif 'kV' in dir:
        voltage = int(dir.split('kV')[0])
    else:
        voltage = int(dir)
    return voltage


def load_px_map():
    path_to_map = r"\\132.187.193.8\junk\sgrischagin\BAD-PIXEL-bin1x1-scans-MetRIC_SCAP_IMGS.tif"
    img = file.image.load(path_to_map)
    return (file.volume.Reader(path_to_map, mode='auto', shape=(1, img.size[0], img.size[1])).load(0)).astype(int)
