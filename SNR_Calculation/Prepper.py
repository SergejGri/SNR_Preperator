import datetime
import os

import SNR_Calculation.curve_db
import SNR_Calculation.curve_db as cdb
from PIL import Image, ImageDraw
import numpy as np
from externe_files import file
from externe_files.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer


class SNRCalculator:
    def __init__(self, img_shape: tuple, header: int, path: str, magnification: float, t_exp: float,
                 res_path: str = None, pixel_size_units: str = None, slice_l: tuple = None, slice_r: tuple = None,
                 watt: float = None, detector_pixel: float = None, smallest_size: float = None, nbins: int = None,
                 exclude: list = None, mode_SNR: bool = True, mode_T: bool = True, SNR_result_name: str = None,
                 T_result_name: str = None):
        """
        see __init__() for the individual types of expected imputs of parameters
        :param img_shape: (detector height, detector width) in pixels
        :param header: image header size in bytes.
        :param path: a string to the path where the raw data (images) from measurement is located
        :param slice_l: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes for the crop area
                for the desired area you want to evaluate (left side)
        :param slice_r: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes into for the
                desired image size you want to evaluate (right side). A slice_r = (0,100)(15,1500)(300,700), for example,
                means an image stack of 100 imgs cropped to the dimension of (1485,400). Even if there is the possibility
                to adjust slice_l and slice_r separately. You should not. Since you want comparable SNR spectra.
        :param watt: adjusted watts for the measurement
        :param t_exp: adjusted exposure time for the measurement in ms
        :param magnification: adjusted magnification for the measurement
        :param detector_pixel: pixel size of the detector. If no value is passed, a default value of 74.8 (current value
                of the MetRIC detector) will be used.
        :param pixel_size_units: a string which should be passed in latex form: $\mu m$ for micrometers etc. If no value
                is passed, default value of micrometer will be used.
        :param filter_mat: a list of used filter material
                wedge there are 4 areas containing 2 steps area1: [4mm,8mm], area2:[12,16]...
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
        self.base_path = path
        self.result_path = res_path
        self.img_shape = img_shape
        if self.img_shape != (1536, 1944):
            print(f'Since your passed detector size of {self.img_shape} differ from my default expectation of '
                  f'(1536, 1944), i do not know which area you want to analyse => You need to pass values for slice_l '
                  f'and slice_r to get plausible results.')
        self.header = header
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

        self.slice_l = slice_l
        if self.slice_l is None:
            self.slice_l = slice(0, 100), slice(60, 1450), slice(105, 855)
        self.slice_r = slice_r
        if self.slice_r is None:
            self.slice_r = slice(0, 100), slice(60, 1450), slice(1080, 1830)
        self.px_map_l = self.slice_l[1], self.slice_l[2]
        self.px_map_r = self.slice_r[1], self.slice_r[2]

        self.x_min = smallest_size
        self.exclude = exclude
        self.mode_SNR = mode_SNR
        self.mode_T = mode_T
        self.SNR_name = SNR_result_name
        self.T_name = T_result_name
        self.px_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'


    def __call__(self, dir, df, dl, dr, area):
        self.from_raw_to_snr(dir, df, dl, dr, area)


    def from_raw_to_snr(self, dir, df, dl, dr, area):
        print(f'Working on dir: {dir} and d: {dl} and {dr}\n')
        results = []

        SNR_eval = SNR_Evaluator()
        filterer_l = ImageSeriesPixelArtifactFilterer()
        filterer_r = ImageSeriesPixelArtifactFilterer()

        res_path_snr, res_path_T = self._get_result_paths(dir, dl, dr)
        str_voltage = dir
        voltage = int(str_voltage.split('_')[0])

        figure = None
        imgs, ref_imgs, dark_imgs = self.prepare_imgs(dir, df, area)
        data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        refs = file.volume.Reader(ref_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        darks = file.volume.Reader(dark_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        if self.mode_T:
            img = (data[self.slice_l] - darks[self.slice_l]) / (refs[self.slice_l] - darks[self.slice_l])
            img = np.mean(img, axis=0)
            self.draw_marker(img, dl)
            T_l = np.min(img, axis=0)
            curve_db = cdb.DB(dl)
            curve_db.add_data(voltage=voltage, d=dl, T=T_l)  # in welche welche Kurve inserten?!
            del img

        if self.mode_SNR:
            SNR_eval.estimate_SNR(data[self.slice_l], refs[self.slice_l], darks[self.slice_l],
                                  exposure_time=self.t_exp,
                                  pixelsize=self.pixel_size,
                                  pixelsize_units=self.pixel_size_units,
                                  series_filterer=filterer_l,
                                  u_nbins=self.nbins,
                                  save_path=os.path.join(res_path_snr,
                                                         f'SNR_{voltage}kV_{dl}_mm_{self.t_exp}ms'))
            figure = SNR_eval.plot(figure, f'{dl} mm')
            results.append(SNR_eval)
        del data, refs, darks

        data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        refs = file.volume.Reader(ref_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        darks = file.volume.Reader(dark_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        if self.mode_T:
            img = (data[self.slice_r] - darks[self.slice_r]) / (refs[self.slice_r] - darks[self.slice_r])
            img = np.mean(img, axis=0)
            self.draw_marker(img, dr)
            T_r = np.min(img, axis=0)
            del img
        if self.mode_SNR:
            SNR_eval.estimate_SNR(data[self.slice_r], refs[self.slice_r], darks[self.slice_r],
                                  exposure_time=self.t_exp,
                                  pixelsize=self.pixel_size,
                                  pixelsize_units=self.pixel_size_units,
                                  series_filterer=filterer_r,
                                  u_nbins=self.nbins,
                                  save_path=os.path.join(res_path_snr,
                                                         f'SNR_{voltage}kV_{dr}_mm_{self.t_exp}ms'))
            figure = SNR_eval.plot(figure, f'{dr} mm')
            results.append(SNR_eval)
        del data, refs, darks

        print(f'Done with {filter}mm filter {dir}: {dl}mm and {dr}mm')

        if self.mode_T:
            self.write_T_data(res_path_T, dl, dr, T_l, T_r, voltage)

        print('finalizing figure...')
        SNR_eval.finalize_figure(figure,
                                 title=f'SNR @ {voltage}kV & {self.watt}W',
                                 smallest_size=self.x_min,
                                 save_path=os.path.join(res_path_snr, f'{voltage}kV_{dl}mm-{dr}mm'))

    def draw_marker(self, img, d):
        _min = np.where(img == np.min(img))
        _miny = int(_min[0])
        _minx = int(_min[1])
        img = Image.fromarray((img * 255).astype(np.uint16))
        r = 15
        draw = ImageDraw.Draw(img)
        leftUpPoint = (_minx - r, _miny - r)
        rightDownPoint = (_minx + r, _miny + r)
        twoPointList = [leftUpPoint, rightDownPoint]
        draw.ellipse(twoPointList, outline='red')
        safe_path = os.path.join(self.result_path, 'Transmission', 'evaluated_images')
        if not os.path.exists(safe_path):
            os.makedirs(safe_path)
        img.save(os.path.join(safe_path, f'{d}mm.tif'))

    def get_dirs(self):
        dirs = []
        for dir in os.listdir(self.base_path):
            if os.path.isdir(os.path.join(self.base_path, dir)) and dir != 'darks':
                if self.exclude is not None:
                    if dir not in self.exclude:
                        dirs.append(dir)
                else:
                    dirs.append(dir)
        return dirs

    def _get_result_paths(self, dir, dl, dr):
        c_date = datetime.datetime.now()
        result_path_snr = None
        result_path_T = None
        if self.mode_SNR:
            if self.SNR_name is None:
                result_path_snr = os.path.join(self.result_path, f'Evaluation_{c_date.year}-{c_date.month}-{c_date.day}', 'SNR', dir)
            else:
                result_path_snr = os.path.join(self.result_path, f'{self.SNR_name}', 'SNR', dir)
            if not os.path.exists(result_path_snr):
                os.makedirs(result_path_snr)
        if self.mode_T:
            if self.T_name is None:
                result_path_T = os.path.join(self.result_path, f'Evaluation_{c_date.year}-{c_date.month}-{c_date.day}', 'Transmission')
            else:
                result_path_T = os.path.join(self.result_path, f'{self.T_name}')
            if not os.path.exists(result_path_T):
                os.makedirs(result_path_T)
        return result_path_snr, result_path_T

    def prepare_imgs(self, dir, df, area):
        imgs = None
        ref_imgs = None
        dark_imgs = None
        if os.path.isdir(os.path.join(self.base_path, dir)) and dir != 'darks':
            dark_imgs = os.path.join(self.base_path, 'darks')
            imgs = os.path.join(self.base_path, dir, 'imgs', df, area)
            ref_imgs = os.path.join(self.base_path, dir, 'refs')
        return imgs, ref_imgs, dark_imgs

    def write_T_data(self, res_path_T, d_l, d_r, T_l, T_r, voltage):
        file_l = f'{d_l}_mm.csv'
        file_r = f'{d_r}_mm.csv'
        print(f'WRITING FILES FOR:\n'
              f'{file_l} \n'
              f'AND \n'
              f'{file_r}')
        with open(os.path.join(res_path_T, file_l), 'a+') as f_l, open(os.path.join(res_path_T, file_r), 'a+') as f_r:
            f_l.write('{};{}\n'.format(voltage, T_l))
            f_r.write('{};{}\n'.format(voltage, T_r))
            f_l.close()
            f_r.close()


'''
    def create_report(self):
        _date = datetime.datetime.now()
        _date = _date.strftime('%x')
        with open(os.path.join(self.result_path, f'{_date}_report.txt'), 'w') as f:
            #f.write(self.__dict__)
            f.close()
'''