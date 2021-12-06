# Copyright 2015-2020 University Würzburg.
#
# Developed at the Lehrstuhl für Röntgenmikroskopie/Universität Würzburg, Josef-Martin-Weg 63, 97074 Würzburg, Germany
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
#    disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
# BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import json
import os

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter

from ext import image, file

version = 1, 2
version_str = f'v{".".join([str(v) for v in version])}'
version_date = '2020-03-04'
print(f'{__name__} version {".".join([str(v) for v in version])} ({version_date})')


try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    def get_color_ls(index):
        color = colors[index % len(colors)]
        ls = ['-', '--', ':'][(index//len(colors)) % 3]
        return color, ls
except ImportError:
    plt = None

VERBOSE = True


#  ======= SNR computations =======
class SNR_Evaluator():
    ''' class to estimate SNR(u) '''

    def __init__(self, data_source=None):
        """

        :param data_source: load results from a file (see load_result())
        """
        self.series_filterer = None

        self.label = None

        self.SNR = None
        self.S = None
        self.N = None

        self.S_ref = None
        self.N_ref = None

        self.S_dark = None
        self.N_dark = None

        self.avg_image = None

        self.data_type = ''
        self.properties = dict()

        if data_source is not None:
            self.load_result(data_source)

    def _get_S_N_SNR(self, images: np.ndarray, u_nbins='auto', apply_log=False, series_filterer=None,
                     use_drift_compensation=False):
        if series_filterer is not None:
            images = series_filterer(images)

        if apply_log:  # neglog and log are effectively identical for power spectra; needs to be done after filter
            np.clip(images, 1e-6, np.inf, out=images)
            np.log(images, out=images)

        K = images.shape[0]
        assert K > 1, 'there must be at least 2 images in a SNR series'
        mean_im = images.mean(axis=0)
        PSC = image.fourier.PowerSpectrumCalculator(mean_im.shape)

        if not use_drift_compensation:
            u, PS_K = PSC.radial_power_spectrum(mean_im, u_nbins)
            u, PS_j = PSC.radial_power_spectrum_avg(images, u_nbins)

            S = (PS_K - PS_j / K) / (1 - 1 / K)
            N = (PS_j - PS_K) / (1 - 1 / K)
            return u, S, N, S / N, mean_im

        else:
            u_nbins = PSC.auto_nbins()
            S, N = np.zeros(u_nbins, 'f4'), np.zeros(u_nbins, 'f4')
            for k in np.arange(K-1):
                images_part = images[k:k+2]
                u, PS_K = PSC.radial_power_spectrum(images_part.mean(axis=0), u_nbins)
                u, PS_j = PSC.radial_power_spectrum_avg(images_part, u_nbins)

                S += 2.*(PS_K - PS_j / 2.)
                N += 2.*(PS_j - PS_K)

            S /= (K-1)
            N /= (K-1)

            return u, S, N, S / N, mean_im

    def estimate_SNR(self, images: np.ndarray, refs: np.ndarray = None, darks: np.ndarray = None,
                     u_nbins='auto', series_filterer=None, apply_log=True, save_path=None, exposure_time=1.,
                     use_drift_compensation=False, pixelsize=1.0, pixelsize_units='px', compute_N0=False):
        '''
        estimate the SNR(u) of an imaging measurement experimental setup from a series of measured images
        the images must contain the same signal/object with different noise realizations

        Note that SNR_Evaluator.N_dark is the noise power spectrum of the subtracted dark image, not the
        noise power spectrum of the dark noise of the image itself. See also SNR_Evaluator.compute_N0().

        great care must be taken that if detector imperfections are present(e.g. dead pixels),
        these must be removed e.g. by passing a ImageSeriesPixelArtifactFilterer as the series_filterer arg
        Note: filtering defects beforehand is less effective if refs/flats are applied

        note that very small SNR cannot be estimated correctly and will instead appear artificially high
        this usually appears in the form of a false lower limit (e.g. at SNR=1e-2) at the higher spatial frequencies
        noise at this lower limit may appear as negative SNR values

        recommendations for measurement parameters
        - 50 images per SNR measurement are recommended. If this is impossible, take as many images as reasonably possible.
        - rule of thumb for exposure time: 1/10th of a realistic exposure time for actual measurements
          (100 x-ray counts per image can be sufficient if the signal is strong enough).
        - Less than 2000 counts in sum (e.g. 20x100) is not recommended.
        - Higher overall exposure times extend the usability of the measurement to higher spatial frequencies;
          if the lower frequencies are sufficient, then short measurements can be used.

        :param images:      measured images (3-dim array with series of 2D images or 4-dim array with series of 3D-images)
        :param refs:        reference/flat images (normalizes intensity without signal to 1), are used for artifact correction
        :param darks:       detector dark images, are used for artifact correction
        :param u_nbins:     number of bins for radially averaged spectra (default 'auto' uses 25 + sqrt(max(image_shape)) )
        :param series_filterer: an object which will remove pixel defects from an image series, callable which takes
                                an images stack and applies its filters (see e.g. ImageSeriesPixelArtifactFilterer)
        :param apply_log:   apply a logarithm if refs were given (should only minimally affect result)
        :param save_path:   save raw data to disk (file endings will be appended to the path given)
        :param exposure_time:   exposure time of the individual images, is used to scale SNR and N
        :param use_drift_compensation:  use an algorithm that is more robust against sample position or source
                                        intensity drift, but noisier
        :param pixelsize:   pixelsize of the measurement; if given, spatial frequency axis will have units of structure size in 1/(2*length)
        :param pixelsize_units: physical units of the pixelsize given, defaults to micrometers (valid LaTeX math code for matplotlib labels)
        :return:            u_bins, S_data, N_data, SNR_data, avg_image
        '''

        if pixelsize_units == 'px':
            print('WARNING: no pixelsize given, spatial frequency units will be meaningless')
        self.properties["pixelsize"] = float(pixelsize)
        self.properties["pixelsize_units"] = pixelsize_units
        self.properties["exposure_time"] = float(exposure_time)
        self.properties["series_shape"] = tuple(int(i) for i in images.shape)
        self.properties["save_path"] = save_path
        self.properties["use_drift_compensation"] = use_drift_compensation

        if refs is None and darks is None:
            self.u, self.S, self.N, self.SNR, self.avg_image = self._get_S_N_SNR(images, u_nbins, False,
                                                                                 series_filterer, use_drift_compensation)
            self.data_type = 'images_only'

        elif darks is None:
            self.properties["refs_shape"] = tuple(int(i) for i in refs.shape)

            data_im = np.nanmean(images, axis=0)
            ref_im = np.nanmean(refs, axis=0)

            series_data_noise = images / ref_im
            self.u, S_data, self.N, SNR_data, self.avg_image = self._get_S_N_SNR(
                series_data_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_data_noise

            series_ref_noise = data_im / refs
            self.u, self.S_ref, self.N_ref, SNR_ref, avg_image_ = self._get_S_N_SNR(
                series_ref_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_ref_noise

            self.S = S_data - self.N_ref / refs.shape[0]
            self.SNR = self.S / self.N
            self.data_type = 'images_refs'

        elif refs is None:
            self.properties["darks_shape"] = tuple(int(i) for i in darks.shape)

            data_im = np.nanmean(images, axis=0)
            dark_im = np.nanmean(darks, axis=0)

            series_data_noise = images - dark_im
            self.u, S_data, self.N, SNR_data, self.avg_image = self._get_S_N_SNR(
                series_data_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_data_noise

            series_dark_noise = data_im - darks
            self.u, self.S_dark, self.N_dark, SNR_dark, avg_image_ = self._get_S_N_SNR(
                series_dark_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_dark_noise

            self.S = S_data - self.N_dark / darks.shape[0]
            self.SNR = self.S / self.N
            self.data_type = 'images_darks'
            if compute_N0:
                self.compute_N0(darks, None, None, series_filterer, apply_log, use_drift_compensation)

        else:
            self.properties["refs_shape"] = tuple(int(i) for i in refs.shape)
            self.properties["darks_shape"] = tuple(int(i) for i in darks.shape)

            if refs.mean() < darks.mean():
                print('WARNING: refs mean is smaller than darks mean, arguments swapped?')

            data_im = np.nanmean(images, axis=0)
            ref_im = np.nanmean(refs, axis=0)
            dark_im = np.nanmean(darks, axis=0)

            series_data_noise = (images - dark_im) / (ref_im - dark_im)
            self.u, S_data, self.N, SNR_data, self.avg_image = self._get_S_N_SNR(
                series_data_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_data_noise

            series_ref_noise = (data_im - dark_im) / (refs - dark_im)
            self.u, self.S_ref, self.N_ref, SNR_ref, avg_image_ = self._get_S_N_SNR(
                series_ref_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_ref_noise

            series_dark_noise = (data_im - darks) / (ref_im - darks)
            self.u, self.S_dark, self.N_dark, SNR_dark, avg_image_ = self._get_S_N_SNR(
                series_dark_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)
            del series_dark_noise

            N_ref_eff = self.N_ref / refs.shape[0]
            N_dark_eff = self.N_dark / darks.shape[0]
            self.S = S_data - N_ref_eff - N_dark_eff
            self.SNR = self.S / self.N
            self.data_type = 'images_refs_darks'
            if compute_N0:
                self.compute_N0(darks, data_im, ref_im, u_nbins, series_filterer, apply_log, use_drift_compensation)

        self.properties["data_type"] = self.data_type

        self.scale_power_spectra(pixelsize, self.avg_image.ndim)
        self.apply_exposure_time(exposure_time)

        if save_path is not None:
            self.save_result(save_path)

        return self.result()

    def scale_power_spectra(self, pixelsize, ndim, advantage=None):
        if pixelsize is not None:
            self.u *= 1/pixelsize

            factor = pixelsize**ndim
            self.S *= factor
            self.N *= factor
            if self.data_type == 'images_refs':
                self.N_ref *= factor
            elif self.data_type == 'images_refs_darks':
                self.N_ref *= factor
                self.N_dark *= factor

    def compute_N0(self, dark_images, image_mean=None, ref_mean=None, u_nbins='auto',
                   series_filterer=None, apply_log=True, use_drift_compensation=False):
        if ref_mean is not None:
            series_N0_noise = (image_mean - dark_images) / (ref_mean - dark_images.mean(axis=0))
        else:
            series_N0_noise = dark_images

        u, self.S0, self.N0, SNR_dark, avg_image_ = self._get_S_N_SNR(
            series_N0_noise, u_nbins, apply_log, series_filterer, use_drift_compensation)

        try:
            self.N0 *= self.properties['pixelsize']**(dark_images.ndim-1)
            if ref_mean is not None:
                self.N0 *= self.properties["exposure_time"]
            else:
                self.N0 /= self.properties["exposure_time"]
        except KeyError:
            print('WARNING: failed to scale N0 to pixelsize and exposure time, run estimate_SNR() before this function')
        return self.N0

    def apply_exposure_time(self, exposure_time):
        if self.data_type == 'images_only':
            self.SNR /= exposure_time
            self.N /= exposure_time

        elif self.data_type == 'images_refs':
            self.SNR /= exposure_time
            self.N *= exposure_time
            self.N_ref *= exposure_time

        elif self.data_type == 'images_refs_darks':
            self.SNR /= exposure_time
            self.N *= exposure_time
            self.N_ref *= exposure_time
            self.N_dark *= exposure_time

    def result(self):
        ''' return the result: u, S, N, SNR, avg_image '''
        return self.u, self.S, self.N, self.SNR, self.avg_image

    def save_result(self, save_name):
        ''' save the result  from estimate_SNR() to a csv file'''
        assert self.SNR is not None, 'no SNR evaluation done to save'
        data = np.vstack((self.u, self.SNR, self.S, self.N)).swapaxes(0, 1)
        assert_base_folder_exists(save_name)
        np.savetxt(save_name + '.txt', data,
                   header="SNR estimate result (radially averaged spectra)\n"
                          "spatial frequency, SNR, signal power spectrum, noise power spectrum\n" +
                   json.dumps(self.properties)
                   )

        file.image.save(-self.avg_image.astype('f4'), save_name + '.tif')

    def load_result(self, save_name):
        SNR = np.genfromtxt(save_name).swapaxes(0, 1)
        self.u, self.SNR, self.S, self.N = SNR
        with open(save_name) as file:
            file.readline()
            file.readline()
            properties = file.readline()
            if properties.startswith('# {'):
                self.properties = json.loads(properties[2:])

    def plot(self, figure: plt.Figure, label=None, only_snr=True):
        '''
        plot the result from estimate_SNR()

        :param figure:  create a new figure if None, else plot into this figure
                        the figure can have 1-3 axes and will plot SNR, S, N if axes are available
                        note: exposure times and pixelsizes of different measurements must be set to get comparable curves
        :param label:   label for data
        :param title:   title for the whole plot
        :param save_path:   path to save fig to (defaults to .pdf)
        :return:
        '''
        if label is None:
            label = self.label
        if label is None:
            raise ValueError('no label given')
        xlabel = f'spatial size [{self.properties["pixelsize_units"]}]'

        if figure is None:
            if only_snr:
                figure, ax = plt.subplots(1, 1, figsize=(5, 5))
                ax.set_xlabel(xlabel)
                ax.set_ylabel('SNR')
            else:
                figure = plt.figure(constrained_layout=True, figsize=(10, 5))
                figure.tight_layout(rect=(0, 0, 1, 1), pad=0.05)
                gs = mpl.gridspec.GridSpec(1, 7, figure=figure)
                axes = figure.add_subplot(gs[:3]), figure.add_subplot(gs[3:5]), figure.add_subplot(gs[5:])
                [ax.set_xlabel(xlabel) for ax in axes]
                [axes[k].set_ylabel(t) for k, t in enumerate(('SNR', 'signal power spectrum', 'noise power spectrum'))]

        if len(figure.axes) == 0:
            if only_snr:
                ax = figure.add_subplot(1, 1, 1)
                ax.set_xlabel(xlabel)
                ax.set_ylabel('SNR')
            else:
                axes = figure.add_subplot(1, 3, 1), figure.add_subplot(1, 3, 2), figure.add_subplot(1, 3, 3)
                [ax.set_xlabel(xlabel) for ax in axes]
                [axes[k].set_ylabel(t) for k, t in enumerate(('SNR', 'signal power spectrum', 'noise power spectrum'))]

        color, ls = get_color_ls(len(figure.axes[0].lines))
        figure.axes[0].semilogy(self.u, self.SNR, label=label, color=color, ls=ls)
        if len(figure.axes) > 1:
            figure.axes[1].semilogy(self.u, self.S, color=color, ls=ls)
        if len(figure.axes) > 2:
            figure.axes[2].plot(self.u, self.N, color=color, ls=ls)

        return figure

    def finalize_figure(self, figure, title=None, save_path=None, smallest_size=None, num_ticks=7):
        if smallest_size is None:
            smallest_size = self.properties["pixelsize"]
        for k, ax in enumerate(figure.axes):
            apply_u_scale(ax, 1, units=self.properties["pixelsize_units"], max_val=1/(2*smallest_size),
                          num_ticks=num_ticks if k == 0 else (num_ticks//2+1))
        figure.axes[0].legend()
        if title is not None:
            figure.suptitle(title)
        if save_path is not None:
            if '.' not in save_path:
                save_path += '.pdf'
            figure.savefig(save_path)


def estimate_SNR(images: np.ndarray, refs: np.ndarray = None, darks: np.ndarray = None,
                 u_nbins='auto', series_filterer=None, apply_log=True, save_path=None,
                 use_drift_compensation=False, pixelsize=1.0):
    return SNR_Evaluator().estimate_SNR(images, refs, darks, u_nbins, series_filterer, apply_log, save_path,
                                        use_drift_compensation=use_drift_compensation, pixelsize=pixelsize)


estimate_SNR.__doc__ = SNR_Evaluator.estimate_SNR.__doc__


#  ======= remove pixel series artifacts =======
class ImageSeriesPixelArtifactFilterer:
    ''' class to filter pixel defects for use with SNR_Estimator '''
    verbose = True

    def __init__(self, filter_defects=True, filter_speckles=True, remove_nonfinite=True, bad_pixel_map=None):
        '''
        remove pixel artifacts from an image series

        filter settings can be set via attributes,
        e.g. ImageSeriesPixelArtifactFilterer.defect_threshold (must be adjusted for non-normalized data)

        can be used as an argument in estimate_SNR()

        if too many defects are filtered, a warning will be printed,
        but no warning will be printed if too few defects are filtered
        (as this cannot be differentiated from from no defects present)

        WARNING: the input data must contain the same signal along the z-axis (=shape[0]),
        e.g. images of the same object but different noise realisations

        :param filter_defects:      filter detector defects, e.g. dead pixels
        :param filter_speckles:     filter speckles (single high intensity noise events)
        :param remove_nonfinite:    remove nonfinite entries
        '''
        self.filter_defects = filter_defects
        self.filter_speckles = filter_speckles
        self.remove_nonfinite = remove_nonfinite
        self.bad_pixel_map = bad_pixel_map

        # default values
        # self.speckle_std_threshold = 5
        # self.speckle_iterations = 2
        # self.speckle_warn_fraction = 0.01

        self.speckle_std_threshold = 2.6
        self.speckle_iterations = 2
        self.speckle_warn_fraction = 0.01

        self.defect_size = 2
        self.defect_threshold = 0.3  # ONLY fitting for [0,1]-data (normalized absorption measurement)
        self.defect_warn_fraction = 0.01

    def __call__(self, image_series: np.ndarray, inplace=False):
        if not inplace:
            image_series = np.copy(image_series)

        mean_image = np.nanmean(image_series, axis=0).astype(image_series.dtype, copy=False)
        loc = ~np.isfinite(mean_image)
        fp = np.array([(0, 0, 1, 0, 0), (0, 1, 0, 1, 0), (1, 0, 1, 0, 1), (0, 0, 1, 0, 0), (0, 1, 0, 1, 0)])
        while np.sum(loc) > 0:
            replace = median_filter(mean_image, footprint=fp, mode='nearest')
            np.copyto(mean_image, replace, where=loc)
            loc = ~np.isfinite(mean_image)

        if self.remove_nonfinite:
            self.remove_nonfinite_series(image_series, mean_image)
        if self.filter_speckles:
            mean_image = np.nanmean(image_series, axis=0).astype(image_series.dtype, copy=False)
            self.filter_speckles_series(image_series, mean_image)
        if self.filter_defects:
            mean_image = np.nanmean(image_series, axis=0).astype(image_series.dtype, copy=False)
            self.filter_defects_series(image_series, mean_image)
        if self.bad_pixel_map is not None:
            medfilt_image = median_filter(mean_image, 5, mode='nearest')[self.bad_pixel_map]
            for k in range(len(image_series)):
                image_series[k][self.bad_pixel_map] = medfilt_image

        return image_series

    def filter_speckles_series(self, images, mean_image):
        frac_sum = 0.
        for j in np.arange(self.speckle_iterations):
            std_image = images.std(axis=0)

            for k in np.arange(images.shape[0]):
                locations = np.logical_or((images[k] > (mean_image + self.speckle_std_threshold*std_image)),
                                          (images[k] < (mean_image - self.speckle_std_threshold*std_image)))
                frac_sum += np.mean(locations)
                np.copyto(images[k], mean_image, where=locations)

        frac_sum /= images.shape[0]

        if frac_sum > self.speckle_warn_fraction:
            print('WARNING: speckle filter replaced too many pixels, increase speckle_std_threshold?')
            print(f'series speckle removal, fraction of replaced pixels: {frac_sum:.4%}')
        elif self.verbose:
            print(f'series speckle removal, fraction of replaced pixels: {frac_sum:.4%}')

    def filter_defects_series(self, images, mean_image):
        mdfilt_mean_image = median_filter(mean_image, self.defect_size*2+1).astype(images.dtype)
        replace_mask = np.abs(mean_image - mdfilt_mean_image) > self.defect_threshold

        for k in np.arange(images.shape[0]):
            np.copyto(images[k, :, :], mdfilt_mean_image, where=replace_mask)

        if np.mean(replace_mask) > self.defect_warn_fraction:
            print(f'WARNING: defect_threshold = {self.defect_threshold:.4e} appears to be set too low')
            print(f'series defect removal, fraction of replaced pixels: {np.mean(replace_mask):.4%}')
        elif self.verbose:
            print(f'series defect removal, fraction of replaced pixels: {np.mean(replace_mask):.4%}')

    def remove_nonfinite_series(self, images, mean_image):
        for k in range(images.shape[0]):
            where = ~np.isfinite(images[k])
            np.copyto(images[k], mean_image, where=where)
        return images

    def test(self, image_series, refs=None, darks=None):
        if refs is not None and darks is not None:
            dark = darks.mean(axis=0)
            image_series = (image_series-dark)/(refs.mean(axis=0), dark)
        elif refs is not None:
            image_series = image_series / refs.mean(axis=0)
        elif darks is not None:
            image_series = image_series - darks.mean(axis=0)

        verbose = self.verbose
        self.verbose = True
        self(image_series, inplace=False)
        self.verbose = verbose


# ====== helpers ======
def detect_lower_SNR_limit(SNR):
    loc = SNR < 0
    if np.any(loc):
        samples = np.copy(SNR[loc])
        m = np.abs(samples.mean())
        samples[::2] *= -1
        low_limit = m+3*samples.std()
        if not np.isfinite(low_limit):
            low_limit = None
        return low_limit


def show_image(image, title=None, figsize=None, vrange=None, perc_val=(1, 99)):
    if figsize is None:
        figsize = 12, 12*image.shape[1]/image.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if vrange is None:
        vrange = np.nanpercentile(image, perc_val)
    ax.imshow(image, cmap=plt.cm.gray, interpolation='none', vmin=vrange[0], vmax=vrange[1])
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
    return fig


def assert_base_folder_exists(fname):
    folder = os.path.split(fname)[0]
    try:
        os.makedirs(folder)
    except (FileExistsError, FileNotFoundError):
        pass


# ======= helpers =======
def generate_test_data(shape=(500, 500), base_intensity=100, nproj=100, nfeatures=100, gauss_blur=0.7,
                       add_pixel_defects=False, feature_max_transparency=0.05):
    test_proj = np.ones(shape, 'f4') * base_intensity
    max_object_size = min(shape) // 5
    for k in range(nfeatures):
        size = np.random.randint(10, max_object_size, size=2)
        origin = np.random.randint(20, test_proj.shape[0] - size[0] - 20), np.random.randint(20, test_proj.shape[1] - size[
            1] - 20)
        test_proj[origin[0]:origin[0] + size[0], origin[1]:origin[1] + size[1]
                  ] *= np.random.randint(int(100*(1-feature_max_transparency)), 100) / 100

    test_proj = gaussian_filter(test_proj, gauss_blur)
    if add_pixel_defects:
        test_proj_d = np.copy(test_proj)
        loc_dead = np.random.randint(0, 1000, size=test_proj.shape) < 1
        test_proj_d[loc_dead] *= 0.3
        loc_bright = np.random.randint(0, 1000, size=test_proj.shape) < 1
        bright_len = len(test_proj[loc_bright])
        bright_values = np.random.poisson(20, size=bright_len) / 100
        test_proj_d[loc_bright] = bright_values
    else:
        test_proj_d = test_proj

    test_projs = test_proj_d[np.newaxis, :, :] * np.ones(nproj, 'f8')[:, np.newaxis, np.newaxis]
    test_projs = np.random.poisson(test_projs).astype('f8')
    return test_projs, test_proj


def apply_u_scale(ax, pixelsize, units='px', max_val=0.5, num_ticks=9, labelevery=1):
    ax.set_xlim(0, max_val)
    u_ticks = np.linspace(0, max_val, num_ticks)
    ax.set_xticks(u_ticks)
    xticklabels = ['$\infty$', ]
    xticklabels += ['{:.2f}'.format(pixelsize/(2*u)) for u in u_ticks[1:]]
    if labelevery > 1:
        xticklabels[::labelevery] = ('',)*len(xticklabels[::labelevery])
    ax.set_xticklabels(xticklabels)
    if units is not None:
        ax.set_xlabel(f'spatial size [{units}]')
    ax.grid()
