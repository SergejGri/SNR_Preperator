import os
import time
import file
import numpy as np
from evaluation.SNR_spectra import ImageSeriesPixelArtifactFilterer, SNR_Evaluator


class Prepper:
    def __init__(self, img_shape: tuple, header: int, path, slice_l: tuple = None, slice_r: tuple = None,
                 watt: float = None, t_exp: float = None, magnification: float = None, detector_pixel: float = None,
                 pixel_size_units=None, filter_mat: list = None, d_pattern: list = None, smallest_size: float = None,
                 exclude: list = None, areas: list = None, mode_SNR: bool = True, mode_T: bool = True,
                 SNR_result_name=None, T_result_name=None):
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
        :param filter_mat: a list of used
        """
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
        self.d_pattern = d_pattern

        self.base_path = path
        self.result_path = r'.'
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
        self.areas = areas
        self.mode_SNR = mode_SNR
        self.mode_T = mode_T
        self.SNR_name = SNR_result_name
        self.T_name = T_result_name
        self.px_map = r'.'

    def __call__(self, dir, d, area, d_l, d_r):
        self.from_raw_to_snr(dir, d, area, d_l, d_r)

    def from_raw_to_snr(self, dir, d, area, d_l, d_r):
        print(f'Working on {dir}')
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



def main():
    start = time.time()

    base_path = r'.'
    bad_pixel_map = r'.'
    watt = 4.5
    filter_materials = ['Al']
    thickness_pattern = ['_0mm Al_', '_1mm Al_', '_2mm Al_']
    areas = ['_1-area_', '_2-area_', '_3-area_', '_4-area_']
    img_shape = (1536, 1944)
    header = 2048

    slice_left = slice(0, 100), slice(50, 1450), slice(92, 859)
    slice_right = slice(0, 100), slice(50, 1450), slice(1076, 1843)
    exposure_time = 1200
    magnification = 21.2781
    detector_pixel = 74.8
    pixel_size_units = '$\mu m$'
    smallest_size = None

    calc_T = True
    calc_SNR = True
    SNR_name = ''
    T_name = ''
    list_exclude = []

    prep_data = Prepper(img_shape, header, path=base_path, watt=watt, t_exp=exposure_time, slice_l=slice_left,
                        slice_r=slice_right, magnification=magnification, detector_pixel=detector_pixel,
                        exclude=list_exclude, smallest_size=smallest_size, pixel_size_units=pixel_size_units,
                        filter_mat=filter_materials, areas=areas, mode_SNR=calc_SNR, mode_T=calc_T,
                        SNR_result_name=SNR_name, T_result_name=T_name)

    dirs = prep_data.get_dirs()
    for dir in dirs:
        for d in range(len(thickness_pattern)):
            d_l, d_r = actual_d(thickness_pattern[d])
            for area in range(len(areas)):
                prep_data(dir, thickness_pattern[d], areas[area], d_l[area], d_r[area])

    print(f'Time: {(time.time() - start) / (60 * 60)} h')


def actual_d(pattern):
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


if __name__ == '__main__':
    main()
