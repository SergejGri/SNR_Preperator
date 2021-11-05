import datetime
import os
import gc
import timeit
from PIL import Image
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
                 overwrite: bool = True, stack_range: tuple = None, t_exp: float = None, pixel_size_units: str = None,
                 _slice: tuple = None, watt: float = None, detector_pixel: float = None, smallest_size: float = None,
                 nbins: int = None, exclude: list = None, mode_SNR: bool = True, params: list = None,
                 mode_T: bool = True, mode_single: bool = False, SNR_result_name: str = None,
                 T_result_name: str = None):
        """
        see __init__() for the individual types of expected imputs of parameters
        :param img_shape: (detector height, detector width) in pixels
        :param header: image header size in bytes.
        :param path: a string to the path where the raw data (images) from measurement is located
        :param _slice: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes for the crop area
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
        :param mode_SNR: default value True. If set to False, only transmission (as long as mode_T==True) will be
                evaluated
        :param mode_T: default value True. Same behavior as mode_SNR
        :param SNR_result_name: your desired name for SNR results
        :param T_result_name: your desired name for Transmission results
        """
        self.path_base = path
        self.path_result = path_result
        self.params = params

        self.img_shape = img_shape
        if self.img_shape != (1536, 1944):
            print(f'Since your passed detector size of {self.img_shape} differ from my default expectation of '
                  f'(1536, 1944), i do not know which area you want to analyse => You need to pass values for slice_l '
                  f'and slice_r to get plausible results.')
        self.header = header
        if stack_range is not None:
            self.stack_range = stack_range
        else:
            self.stack_range = (0, 200)
        self.M = magnification
        self.t_exp = t_exp
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

        self.path_px_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'
        self._slice = _slice
        if self._slice is None:
            # Tiefe, Höhe, Breite
            self._slice = slice(self.stack_range[0], self.stack_range[-1] - self.stack_range[0]), \
                          slice(60,1460), \
                          slice(125, 1825)
        self.px_map_slice = self._slice[1], self._slice[2]

        self.x_min = smallest_size
        self.exclude = exclude
        self.mode_SNR = mode_SNR
        self.mode_T = mode_T
        if not self.mode_T: print('mode_T = False --> No Transmission calc.')
        self.c_date = datetime.datetime.now()

        self.mode_single = mode_single
        if SNR_result_name is not None: self.SNR_name = SNR_result_name
        self.SNR_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_SNR'
        if T_result_name is not None: self.T_name = T_result_name
        self.T_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_T'

    def __call__(self, dir: str, subf: list):
        if not self.mode_single:
            self.snr_multi(dir, subf)
        else:
            print('coming soon')

    def snr_multi(self, dir, subf):
        path_save_SNR = os.path.join(self.path_result, self.SNR_name, dir)
        path_save_T = os.path.join(self.path_result, self.T_name)
        if not os.path.exists(path_save_T):
            os.makedirs(path_save_T)

        SNR_eval = SNR_Evaluator()
        str_voltage = get_voltage(dir)

        results = []
        figure = None
        for a in subf:
            _d = self.filter_area_to_t(a)
            imgs, imgs_ref, imgs_dark = self.prepare_imgs(self.path_base, dir, a)

            print(f'working on {dir}: {_d} mm')
            data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header,
                                      dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))
            darks = file.volume.Reader(imgs_dark, mode='raw', shape=self.img_shape, header=self.header,
                                       dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))
            refs = file.volume.Reader(imgs_ref, mode='raw', shape=self.img_shape, header=self.header,
                                      dtype='u2').load_range((self.stack_range[0], self.stack_range[-1]))

            if self.mode_T:
                T = self.calc_T(data, refs, darks)
                self.write_T_data(path_save_T, _d, T, str_voltage)

            if self.mode_SNR:
                self.t_exp = get_t_exp(imgs)
                SNR_eval, figure = self.calc_SNR(SNR_eval, imgs, path_save_SNR, figure, data, refs, darks, _d, str_voltage)
                figure = SNR_eval.plot(figure, f'{_d} mm')
                results.append(SNR_eval)
            del data
            gc.collect()
            print(f'Done with {dir}: {_d} mm')

        print('finalizing figure...')
        SNR_eval.finalize_figure(figure, title=f'{dir}_@{self.watt}W', smallest_size=self.x_min,
                                 save_path=os.path.join(path_save_SNR, f'{str_voltage}kV'))

    def write_T_data(self, path_T, d, T, voltage):
        file_l = f'{d}_mm.csv'
        print(f'WRITING FILES {d} mm')

        _path_T = os.path.join(path_T, file_l)

        with open(os.path.join(_path_T), 'a+') as f_l:
            f_l.write('{};{}\n'.format(voltage, T))
            f_l.close()

    def calc_T(self, data, refs, darks):
        img = (data[self._slice] - darks[self._slice]) / (refs[self._slice] - darks[self._slice])
        T = img[np.where(img > 0)].mean()
        del img
        gc.collect()
        return T

    def calc_SNR(self, snr_obj, path_to_files, path_save_SNR, figure, data, refs, darks, _d, voltage):
        px_map = load_px_map()
        filterer = ImageSeriesPixelArtifactFilterer()
        # filterer = ImageSeriesPixelArtifactFilterer(bad_pixel_map=px_map[self.px_map_slice])
        self.t_exp = get_t_exp(path_to_files)

        snr_obj.estimate_SNR(data[self._slice], refs[self._slice], darks[self._slice], exposure_time=self.t_exp,
                             pixelsize=self.pixel_size, pixelsize_units=self.pixel_size_units,
                             series_filterer=filterer, u_nbins=self.nbins,
                             save_path=os.path.join(path_save_SNR,
                                                    f'SNR(u)-{self.watt}_W-{voltage}_kV-{_d}_mm-expTime_{self.t_exp}'))
        return snr_obj, figure

    @staticmethod
    def prepare_imgs(path, dir, subf):
        imgs = None
        ref_imgs = None
        dark_imgs = None
        if os.path.isdir(os.path.join(path, dir)):
            dark_imgs = os.path.join(path, dir, 'darks')
            imgs = os.path.join(path, dir, str(subf))
            ref_imgs = os.path.join(path, dir, 'refs')
        return imgs, ref_imgs, dark_imgs

    @staticmethod
    def filter_area_to_t(thick):
        return int(thick)

    @staticmethod
    def get_dirs(pathh, exclude):
        list_dir = []
        for dirr in os.listdir(pathh):
            if os.path.isdir(os.path.join(pathh, dirr)) and dirr not in exclude and dirr != 'darks':
                list_dir.append(dirr)
        return list_dir


def get_t_exp(path):
    t_exp = None
    for file in os.listdir(path):

        piece_l = file.split('expTime_')[1]
        piece_r = piece_l.split('__')[0]
        t_exp = int(piece_r)
        t_exp = t_exp / 1000
        break
    return t_exp


def get_voltage(dir):
    if '_' in dir:
        voltage = int(dir.split('_')[0])
    elif 'kV' in dir:
        voltage = int(dir.split('kV')[0])
    else:
        voltage = int(dir)
    return voltage


def load_px_map():
    path_to_map = r"C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif"
    img = file.image.load(path_to_map)
    return (file.volume.Reader(path_to_map, mode='auto', shape=(1, img.size[0], img.size[1])).load(0)).astype(int)
