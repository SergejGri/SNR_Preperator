import os
import numpy as np
import file
from evaluation.SNR_spectra import ImageSeriesPixelArtifactFilterer, SNR_Evaluator


class SNRCalculator:
    def __init__(self, img_shape: tuple, header: int, path, slice_l: tuple = None, slice_r: tuple = None,
                 watt: float = None, t_exp: float = None, magnification: float = None, detector_pixel: float = None,
                 pixel_size_units=None, filter_mat: list = None, smallest_size: float = None, nbins: int = None,
                 exclude: list = None, mode_SNR: bool = True, mode_T: bool = True, SNR_result_name=None,
                 T_result_name=None):
        """
        see __init__() for the individual types of expected imputs of parameters
        :param img_shape: (image height, image width) in pixels
        :param header: image header size in bytes.
        :param path: a string to the path where the raw data from measurement is located
        :param slice_l: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes into for the
        desired image size you want to evaluate (left side)
        :param slice_r: (axis=0/stack size)(axis=1/height)(axis=2/width) pass for each axis the sizes into for the
        desired image size you want to evaluate (right side). A slice_r = (0,100)(15,1500)(300,700), for example, means
        that the image stack is cropped from 0 to 99 images with images size of (1485,400). Even if there is the
        possibility to adjust slice_l and slice_r. You should not. Since you want comparable SNR spectra.
        :param watt: adjusted watts for the measurement
        :param t_exp: adjusted exposure time for the measurement
        :param magnification: adjusted magnification for the measurement
        :param detector_pixel: pixel size of the detector
        :param pixel_size_units: a string which should be passed in latex form $\mu m$ for micrometers etc.
        :param filter_mat: a list of used filter material
        wedge there are 4 areas containing 2 steps area1: [4mm,8mm], area2:[12,16]...
        :param smallest_size: if smallest size is passed, the maximal value of the u-axis will be set to the passed
        value
        :param exclude: expects a list of kV-folders which the user want to avoid. IF no list ist passed, the class will
        evaluate all voltage folders in the base_path
        :param mode_SNR: default value True. If set the False, only transmission (as long as mode_T=True) will be
        evaluated
        :param mode_T: default value True. Same behavior as mode_SNR
        :param SNR_result_name: your desired name for SNR results
        :param T_result_name: your desired name for SNR results
        """
        if nbins is not None:
            self.nbins = nbins
        else:
            self.nbins = 'auto'
        self.watt = watt
        self.img_shape = img_shape
        self.header = header
        self.t_exp = t_exp
        self.M = magnification
        self.det_pixel = detector_pixel
        if detector_pixel is not None and magnification is not None:
            self.pixel_size = self.det_pixel / self.M
        else:
            print('Could not estimate pixel size, since no parameter for detector pixel and magnification were passe')
        self.pixel_size_units = pixel_size_units

        self.filter_mat = filter_mat
        if len(self.filter_mat) < 2:
            self.filter_mat = filter_mat[0]

        self.base_path = path
        self.result_path = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR'
        if slice_l is None:
            self.slice_l = slice(0, 100), slice(50, 1450), slice(92, 859)
        self.slice_l = slice_l
        self.px_map_l = self.slice_l[1], self.slice_l[2]
        if slice_r is None:
            self.slice_r = slice(0, 100), slice(50, 1450), slice(1076, 1843)
        self.slice_r = slice_r
        self.px_map_r = self.slice_r[1], self.slice_r[2]
        self.x_min = smallest_size
        self.exclude = exclude
        self.mode_SNR = mode_SNR
        self.mode_T = mode_T
        self.SNR_name = SNR_result_name
        self.T_name = T_result_name
        self.px_map = r'C:\Users\Sergej Grischagin\Desktop\Auswertung_SNR\BAD-PIXEL-bin1x1-scans-MetRIC.tif'

    def __call__(self, dir, d, area, d_l, d_r):
        self.from_raw_to_snr(dir, d, area, d_l, d_r)

    def from_raw_to_snr(self, dir, d, area, d_l, d_r):
        print(f'Working on: \n'
              f'dir: {dir} \n'
              f'd: {d}')
        results = []

        SNR_eval = SNR_Evaluator()
        filterer_l = ImageSeriesPixelArtifactFilterer()
        filterer_r = ImageSeriesPixelArtifactFilterer()

        res_path_snr, res_path_T = self.get_result_paths(dir, d)
        str_voltage = dir
        voltage = int(str_voltage.split('_')[0])

        figure = None

        imgs, ref_imgs, dark_imgs = self.prepare_imgs(dir, d, area)
        data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        refs = file.volume.Reader(ref_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        darks = file.volume.Reader(dark_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        if self.mode_T:
            img = (data[self.slice_l] - darks[self.slice_l]) / (refs[self.slice_l] - darks[self.slice_l])
            T_l = np.min(np.mean(img, axis=0))
            del img
        SNR_eval.estimate_SNR(data[self.slice_l], refs[self.slice_l], darks[self.slice_l],
                              exposure_time=self.t_exp,
                              pixelsize=self.pixel_size,
                              pixelsize_units=self.pixel_size_units,
                              series_filterer=filterer_l,
                              u_nbins=self.nbins,
                              save_path=os.path.join(res_path_snr,
                                                     f'SNR_{voltage}kV_{d_l}_mm_{self.t_exp}ms'))
        figure = SNR_eval.plot(figure, f'{d_l} mm')
        results.append(SNR_eval)
        del data, refs, darks

        data = file.volume.Reader(imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        refs = file.volume.Reader(ref_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        darks = file.volume.Reader(dark_imgs, mode='raw', shape=self.img_shape, header=self.header).load_all()
        if self.mode_T:
            img = (data[self.slice_r] - darks[self.slice_r]) / (refs[self.slice_r] - darks[self.slice_r])
            T_r = np.min(np.mean(img, axis=0))
            del img
        SNR_eval.estimate_SNR(data[self.slice_r], refs[self.slice_r], darks[self.slice_r],
                              exposure_time=self.t_exp,
                              pixelsize=self.pixel_size,
                              pixelsize_units=self.pixel_size_units,
                              series_filterer=filterer_r,
                              u_nbins=self.nbins,
                              save_path=os.path.join(res_path_snr,
                                                     f'SNR_{voltage}kV_{d_r}_mm_{self.t_exp}ms'))
        figure = SNR_eval.plot(figure, f'{d_r} mm')
        results.append(SNR_eval)
        del data, refs, darks

        print(f'Done with {filter}mm filter \n'
              f'{dir}: {d_l}mm and {d_r}mm')

        if self.mode_T:
            self.write_T_data(res_path_T, d_l, d_r, T_l, T_r, voltage)

        print('finalizing figure...')
        SNR_eval.finalize_figure(figure,
                                 title=f'SNR @ {voltage}kV & {self.watt}W',
                                 smallest_size=self.x_min,
                                 save_path=os.path.join(res_path_snr, f'{voltage}kV_{d_l}mm-{d_r}mm'))

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

    def get_result_paths(self, dir, d):
        result_path_snr = os.path.join(self.result_path, f'{self.SNR_name}', dir, f'{d}')
        if not os.path.exists(result_path_snr):
            os.makedirs(result_path_snr)
        result_path_T = os.path.join(self.result_path, f'{self.T_name}')
        if not os.path.exists(result_path_T):
            os.makedirs(result_path_T)
        return result_path_snr, result_path_T

    def mat_naming(self, pattern):
        if pattern == '_0mm Al_':
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

    def prepare_imgs(self, dir, d, area):
        imgs = None
        ref_imgs = None
        dark_imgs = None
        if os.path.isdir(os.path.join(self.base_path, dir)) and dir != 'darks':
            dark_imgs = os.path.join(self.base_path, 'darks')
            imgs = os.path.join(self.base_path, dir, 'imgs', d, area)
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

    def actual_d(self, pattern):
        if pattern == '_0mm Al_':
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
