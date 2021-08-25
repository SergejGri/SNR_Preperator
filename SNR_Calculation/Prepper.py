import datetime
import os
import gc
import SNR_Calculation.curve_db
import SNR_Calculation.curve_db as cdb
from PIL import Image, ImageDraw
import numpy as np
from externe_files import file
from externe_files.SNR_spectra import SNR_Evaluator, ImageSeriesPixelArtifactFilterer


class SNRCalculator:
    def __init__(self, img_shape: tuple, header: int, path: str, path_result: str, magnification: float,
                 overwrite: bool = False, t_exp: float = None, pixel_size_units: str = None, slice_l: tuple = None,
                 slice_r: tuple = None, watt: float = None, detector_pixel: float = None, smallest_size: float = None,
                 nbins: int = None, exclude: list = None, mode_SNR: bool = True, mode_T: bool = True,
                 mode_single: bool = False, SNR_result_name: str = None, T_result_name: str = None):
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
        self.path_base = path
        self.path_result = path_result
        if os.path.exists(self.path_result):
            print('Caution! Given path is not empty while overwrite parameter is set to False.')
        else:
            os.makedirs(self.path_result)

        self.img_shape = img_shape
        if self.img_shape != (1536, 1944):
            print(f'Since your passed detector size of {self.img_shape} differ from my default expectation of '
                  f'(1536, 1944), i do not know which area you want to analyse => You need to pass values for slice_l '
                  f'and slice_r to get plausible results.')
        self.header = header
        self.stack_range = (29, 99)
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
            self.slice_l = slice(29, 99), slice(130, 830), slice(250, 750)
        self.slice_r = slice_r
        if self.slice_r is None:
            self.slice_r = slice(29, 99), slice(130, 830), slice(1200, 1700)
        self.px_map_l = self.slice_l[1], self.slice_l[2]
        self.px_map_r = self.slice_r[1], self.slice_r[2]

        self.x_min = smallest_size
        self.exclude = exclude
        self.mode_SNR = mode_SNR
        self.mode_T = mode_T
        self.c_date = datetime.datetime.now()

        self.mode_single = mode_single
        if SNR_result_name is not None:
            self.SNR_name = SNR_result_name
        self.SNR_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_SNR'
        if T_result_name is not None:
            self.T_name = T_result_name
        self.T_name = f'{self.c_date.year}-{self.c_date.month}-{self.c_date.day}_T'
        self.px_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'

    def __call__(self, dir, df):
        if not self.mode_single:
            self.snr_multi(dir, df)
        else:
            print('coming soon')

    def snr_multi(self, dir, df):
        rng = (29, 99)
        results = []

        imgs_dark = r'\\132.187.193.8\junk\sgrischagin\2021-08-09-Sergej_SNR_Stufelkeil_40-75kV\darks'
        _, imgs_ref, _ = self.prepare_imgs(self.path_base, dir, df)
        path_save_SNR = os.path.join(self.path_result, self.SNR_name, dir)
        path_save_T = os.path.join(self.path_result, self.T_name)
        if not os.path.exists(path_save_T):
            os.makedirs(path_save_T)

        SNR_eval = SNR_Evaluator()
        filterer_l = ImageSeriesPixelArtifactFilterer()
        filterer_r = ImageSeriesPixelArtifactFilterer()


        str_voltage = dir
        if '_' in str_voltage:
            voltage = int(str_voltage.split('_')[0])
        else:
            voltage = int(str_voltage.split('kV')[0])

        areas = self.get_areas()

        darks = file.volume.Reader(imgs_dark, mode='raw', shape=self.img_shape, header=self.header).load_range(self.stack_range)
        refs = file.volume.Reader(imgs_ref, mode='raw', shape=self.img_shape, header=self.header).load_range(self.stack_range)
        figure = None
        for a in range(len(areas)):
            d_l, d_r = self.filter_area_to_t(df, areas[a])

            imgs, _, _ = self.prepare_imgs(self.path_base, dir, df, areas[a])

            data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header).load_range(self.stack_range)

            if self.mode_T:
                img = (data[self.slice_l] - darks[self.slice_l]) / (refs[self.slice_l] - darks[self.slice_l])
                img = np.mean(img, axis=0)
                T_l = np.min(img)

                img = (data[self.slice_r] - darks[self.slice_r]) / (refs[self.slice_r] - darks[self.slice_r])
                img = np.mean(img, axis=0)
                T_r = np.min(img)

                self.write_T_data(path_save_T, d_l, d_r, T_l, T_r, voltage)
                del img
                gc.collect()

            if self.mode_SNR:
                self.t_exp = self._get_t_exp(imgs)

                SNR_eval.estimate_SNR(data[self.slice_l], refs[self.slice_l], darks[self.slice_l],
                                      exposure_time=self.t_exp,
                                      pixelsize=self.pixel_size,
                                      pixelsize_units=self.pixel_size_units,
                                      series_filterer=filterer_l,
                                      u_nbins=self.nbins,
                                      save_path=os.path.join(path_save_SNR, f'SNR_{voltage}kV_{d_l}_mm_expTime_{self.t_exp}'))
                figure = SNR_eval.plot(figure, f'{d_l} mm')
                results.append(SNR_eval)

                SNR_eval.estimate_SNR(data[self.slice_r], refs[self.slice_r], darks[self.slice_r],
                                      exposure_time=self.t_exp,
                                      pixelsize=self.pixel_size,
                                      pixelsize_units=self.pixel_size_units,
                                      series_filterer=filterer_r,
                                      u_nbins=self.nbins,
                                      save_path=os.path.join(path_save_SNR, f'SNR_{voltage}kV_{d_r}_mm_expTime_{self.t_exp}'))
                figure = SNR_eval.plot(figure, f'{d_r} mm')
                results.append(SNR_eval)

            del data
            gc.collect()
            print(f'Done with {df}mm filter and {dir}: {d_l}mm and {d_r}mm')

        print('finalizing figure...')
        SNR_eval.finalize_figure(figure,
                                 title=f'SNR @ {voltage}kV & {self.watt}W',
                                 smallest_size=self.x_min,
                                 save_path=os.path.join(path_save_SNR, f'{voltage}kV_{df}'))

    def write_T_data(self, path_T, d_l, d_r, T_l, T_r, voltage):
        file_l = f'{d_l}_mm.csv'
        file_r = f'{d_r}_mm.csv'
        print(f'WRITING FILES FOR:\n'
              f'{file_l} \n'
              f'AND \n'
              f'{file_r}')

        _path_T_l = os.path.join(path_T, file_l)
        _path_T_r = os.path.join(path_T, file_r)

        with open(os.path.join(_path_T_l), 'a+') as f_l, open(os.path.join(_path_T_r), 'a+') as f_r:
            f_l.write('{};{}\n'.format(voltage, T_l))
            f_r.write('{};{}\n'.format(voltage, T_r))
            f_l.close()
            f_r.close()

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

    def _get_result_paths(self, dir, dl, dr):
        defaul_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR'
        result_path_snr = None
        result_path_T = None
        if self.mode_SNR:
            if self.SNR_name is None:
                result_path_snr = os.path.join(self.result_path, f'Evaluation_{self.c_date.year}-{self.c_date.month}-{self.c_date.day}', 'SNR', dir)
            else:
                result_path_snr = os.path.join(self.result_path, f'{self.SNR_name}', 'SNR', dir)
            if not os.path.exists(result_path_snr):
                os.makedirs(result_path_snr)
        if self.mode_T:
            if self.T_name is None:
                result_path_T = os.path.join(self.result_path, f'Evaluation_{self.c_date.year}-{self.c_date.month}-{self.c_date.day}', 'Transmission')
            else:
                result_path_T = os.path.join(self.result_path, f'{self.T_name}')
            if not os.path.exists(result_path_T):
                os.makedirs(result_path_T)
        return result_path_snr, result_path_T


    def _get_t_exp(self, path):
        t_exp = None
        for file in os.listdir(path):
            piece_l = file.split('expTime_')[1]
            piece_r = piece_l.split('__')[0]
            t_exp = int(piece_r)
            break
        return t_exp

    @staticmethod
    def prepare_imgs(path, dir, df, area=None):
        imgs = None
        ref_imgs = None
        dark_imgs = None
        if os.path.isdir(os.path.join(path, dir)) and dir != 'darks':
            dark_imgs = os.path.join(path, 'darks')
            if area is not None:
                imgs = os.path.join(path, dir, 'imgs', df, area)
            ref_imgs = os.path.join(path, dir, 'refs')
        return imgs, ref_imgs, dark_imgs

    @staticmethod
    def get_d():
        thick_0 = [4, 8, 12, 16, 20, 24, 28, 32]
        thick_1 = [5, 9, 13, 17, 21, 25, 29, 33]
        thick_2 = [6, 10, 14, 18, 22, 26, 30, 34]
        thicknesses = [thick_0, thick_1, thick_2]
        return thicknesses

    @staticmethod
    def get_df():
        list_df = ['_none_', '_1mm Al_', '_2mm Al_']
        return list_df

    @staticmethod
    def get_areas():
        list_areas = ['_1-area_', '_2-area_', '_3-area_', '_4-area_']
        return list_areas

    @staticmethod
    def current_d(pattern):
        if pattern == '_none_':
            list_left_0mm = ['4', '12', '20', '28']
            list_right_0mm = ['8', '16', '24', '32']
            return list_left_0mm, list_right_0mm
        if pattern == '_1mm Al_':
            list_left_1mm = ['5', '13', '21', '29']
            list_right_1mm = ['9', '17', '25', '33']
            return list_left_1mm, list_right_1mm
        if pattern == '_2mm Al_':
            list_left_2mm = ['6', '14', '22', '30']
            list_right_2mm = ['10', '18', '26', '34']
            return list_left_2mm, list_right_2mm

    @staticmethod
    def filter_area_to_t(thick, area):
        if thick == '_none_':
            if area == '_1-area_':
                return 4, 8
            elif area == '_2-area_':
                return 12, 16
            elif area == '_3-area_':
                return 20, 24
            else:
                return 28, 32
        elif thick == '_1mm Al_':
            if area == '_1-area_':
                return 5, 9
            elif area == '_2-area_':
                return 13, 17
            elif area == '_3-area_':
                return 21, 25
            else:
                return 29, 33
        else:
            if area == '_1-area_':
                return 6, 10
            elif area == '_2-area_':
                return 14, 18
            elif area == '_3-area_':
                return 22, 26
            else:
                return 30, 34

    @staticmethod
    def get_dirs(pathh, exclude):
        list_dir = []
        for dirr in os.listdir(pathh):
            if os.path.isdir(os.path.join(pathh, dirr)) and dirr not in exclude and dirr != 'darks':
                list_dir.append(dirr)
        return list_dir