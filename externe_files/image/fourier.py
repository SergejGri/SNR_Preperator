''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import numpy as np

try:
    import pyfftw, os
    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(60)
    FFT_THREADS = os.cpu_count()
except ImportError:
    FFT_THREADS = None
    pyfftw = None


# ==== Fourier analysis and filters ====
class FourierFilterer:
    planner_effort = 'FFTW_ESTIMATE'

    def __init__(self, imshape=None, dtype='f8', batch=True, pad_for_speed=True,
                 pad_length=0, pad_mode='reflect', pad_constant_value=0.0):
        '''
        compute fourier transforms and filters in 1-3 dimensions

        :param imshape:           shape of the array
        :param dtype:
        :param pad_length:
        :param pad_mode:
        :param pad_constant_value:
        '''
        self._use_fftw = pyfftw is not None
        self.__batch = batch
        self.batch = batch
        self.pad_for_speed = pad_for_speed
        self.pad_mode = pad_mode
        self.pad_constant_value = pad_constant_value
        self.input_shape = imshape
        self.dtype = dtype
        self._pad_length = pad_length

        self.Ffilter = None

        self.input_shape = ()
        if imshape is not None:
            self.prepare_shape(imshape)

    def prepare_shape(self, shape=None):
        #print('prepare_shape', len(shape), len(self.input_shape), np.array(shape), self.input_shape)
        if len(shape) != len(self.input_shape) or not np.all(np.array(shape) == self.input_shape):
            self.input_shape = np.array(shape)
            if self.pad_for_speed:
                self.padded_shape = tuple([self._get_fast_fft_length(length + 2 * self.pad_length) for length in self.input_shape])
            else:
                self.padded_shape = tuple([length + 2 * self.pad_length for length in self.input_shape])

            self.ft_shape = tuple(np.prod(arr.shape) for arr in self.u_axes())

            if self._use_fftw:
                self.arr = pyfftw.zeros_aligned(self.padded_shape, self.dtype) + self.pad_constant_value
            else:
                self.arr = np.zeros(self.padded_shape, self.dtype) + self.pad_constant_value

        self.pad_lengths = []
        for k in range(len(self.input_shape)):
            before = (self.padded_shape[k] - self.input_shape[k]) // 2
            after = self.padded_shape[k] - self.input_shape[k] - before
            self.pad_lengths.append((before, after))
        self.pad_slices = tuple(slice(tup[0], self._slice_invert(tup[1])) for tup in self.pad_lengths)

    @property
    def pad_length(self):
        return self._pad_length
    
    @pad_length.setter
    def pad_length(self, pad_length):
        self._pad_length = pad_length

    def add_pad_length(self, length):
        if self.pad_length == 0:
            self.pad_length = length
        else:
            self.pad_length = sum_pad_ranges((self.pad_length, length))
            
    # padding methods
    @staticmethod
    def _get_fast_fft_length(min_length, other_radices=(3, 5), min_2pot=3):
        '''
        chooses the fft size as a multiple of 2, 3, and 5 to make use of the fft implementation radices
        gives very reliable fft speeds typically up to 2x faster than computing the original shape
        this usually makes some padding necessary

        the observed speed gain in extreme cases is 40x (very slow fft on the original shape)(22, 508, 700)
        only for 'FFTW_ESTIMATE' -- using this method makes 'FFTW_MEASURE' unnecessary

        note that a rfft is only valid if the length of the last axis is an even number,
        which this function automatically enforces

        :param min_length:      actual length of the image
        :param other_radices:   fft radices implemented (in addition to 2)
        :param min_2pot:        minimum exponent for radix 2 (default: use multiple of 8 to increase speed)
        :return:
        '''
        max_length = 2**int(np.ceil(np.log2(min_length)+1))
        radix_conributions = []
        radix_conributions.append(2 ** np.arange(min_2pot, int(np.log(max_length) / np.log(2)) + 1))
        for k, radix in enumerate(other_radices):
            radix_conributions.append(radix ** np.arange(0, int(np.log(max_length / 2 ** min_2pot) / np.log(radix)) + 1))

        n = 1 + len(other_radices)
        all_sizes = np.prod([radix_conributions[k][PowerSpectrumCalculator._nd_newaxis_other(k, n)] for k in np.arange(n)]).astype('i8')
        all_sizes = all_sizes[np.logical_and(np.greater_equal(all_sizes, min_length), np.less_equal(all_sizes, max_length))]
        return all_sizes.min()

    @staticmethod
    def _nd_newaxis_other(k, ndim):
        return tuple(slice(None) if m == k else np.newaxis for m in np.arange(ndim))

    @staticmethod
    def _slice_invert(val):
        if val == 0:
            return None
        elif val is not None:
            return -val
        else:
            return val

    @property
    def batch(self):
        return self.__batch

    @batch.setter
    def batch(self, val):
        self.__batch = val
        if val:
            self.planner_effort = 'FFTW_MEASURE'
        else:
            self.planner_effort = 'FFTW_ESTIMATE'

    def _pad(self, arr_in, for_power_spectrum=False):
        if self.pad_mode == 'constant':
            self._pad_constant(arr_in)
        elif self.pad_mode == 'reflect':
            self._pad_reflect(arr_in)
        else:
            raise ValueError(f'invalid pad mode: {self.pad_mode}')

    def _pad_constant(self, arr_in):
        self.arr[self.pad_slices] = arr_in

    def _pad_reflect(self, arr_in):
        self.arr[self.pad_slices] = arr_in

        for axis in np.arange(arr_in.ndim):
            if self.pad_lengths[axis][0] > 0:
                pad_before = min(self.pad_lengths[axis][0], arr_in.shape[axis] - 1)
                rest_before = self.pad_lengths[axis][0] - pad_before
                self._copy_range(self.arr, axis, (self.pad_lengths[axis][0] + 1, self.pad_lengths[axis][0] + pad_before + 1),
                                 (rest_before, pad_before + rest_before), True)

            if self.pad_lengths[axis][1] > 0:
                pad_after = min(self.pad_lengths[axis][1], arr_in.shape[axis] - 1)
                rest_after = self.pad_lengths[axis][1] - pad_after
                self._copy_range(self.arr, axis, (-pad_after - self.pad_lengths[axis][1] - 1, -self.pad_lengths[axis][1] - 1),
                                 (-pad_after - rest_after, None if rest_after == 0 else -rest_after), True)

    @staticmethod
    def _copy_range(arr, axis=0, range_r=(None, None), range_w=(None, None), invert_read=False):
        all_slice = slice(None, None)
        slices_r = tuple(all_slice if k != axis else slice(range_r[0], range_r[1]) for k in range(arr.ndim))
        slices_w = tuple(all_slice if k != axis else slice(range_w[0], range_w[1]) for k in range(arr.ndim))

        if invert_read:
            invert_slice = tuple(all_slice if k != axis else slice(None, None, -1) for k in range(arr.ndim))
            arr[slices_w] = arr[slices_r][invert_slice]
        else:
            arr[slices_w] = arr[slices_r]

    def _unpad(self, arr):
        return arr[self.pad_slices]

    # fft functions
    def rfftn(self, arr, for_power_spectrum=False):
        self.prepare_shape(arr.shape)
        self._pad(arr, for_power_spectrum=for_power_spectrum)
        if self._use_fftw:
            result_ft = pyfftw.interfaces.numpy_fft.rfftn(self.arr, threads=FFT_THREADS,
                                                          planner_effort=self.planner_effort)
        else:
            result_ft = np.fft.rfftn(self.arr)
        return result_ft

    def irfftn(self, arr_ft):
        if self._use_fftw:
            result = pyfftw.interfaces.numpy_fft.irfftn(arr_ft, threads=FFT_THREADS,
                                                        planner_effort=self.planner_effort)
        else:
            result = np.fft.irfftn(arr_ft)
        return self._unpad(result)

    def fftn(self, arr, for_power_spectrum=False):
        self.prepare_shape(arr.shape)
        self._pad(arr, for_power_spectrum=for_power_spectrum)
        if self._use_fftw:
            result_ft = pyfftw.interfaces.numpy_fft.fftn(self.arr, threads=FFT_THREADS,
                                                          planner_effort=self.planner_effort)
        else:
            result_ft = np.fft.fftn(self.arr)
        return result_ft

    def ifftn(self, arr_ft):
        if self._use_fftw:
            result = pyfftw.interfaces.numpy_fft.ifftn(arr_ft, threads=FFT_THREADS,
                                                        planner_effort=self.planner_effort)
        else:
            result = np.fft.ifftn(arr_ft)
        return self._unpad(result)

    def set_filter(self, Ffilter):
        self.Ffilter = Ffilter

    def apply_filter(self, arr, filter=None, copy_mem=True):
        arr_ft = self.rfftn(arr)
        if filter is not None:
            arr_ft *= filter
        elif self.Ffilter is not None:
            arr_ft *= self.Ffilter
        else:
            raise ValueError('must give filter either via function argument for apply_filter() or via set_filter()')
        if copy_mem:
            return np.copy(self.irfftn(arr_ft))
        else:
            return self.irfftn(arr_ft)

    # spatial frequency functions
    def u_axes(self):
        if len(self.padded_shape) == 3:
            u_z = np.fft.fftfreq(self.padded_shape[0])[:, np.newaxis, np.newaxis]
            u_y = np.fft.fftfreq(self.padded_shape[1])[np.newaxis, :, np.newaxis]
            u_x = np.fft.rfftfreq(self.padded_shape[2])[np.newaxis, np.newaxis, :]
            return u_z, u_y, u_x
        if len(self.padded_shape) == 2:
            u_y = np.fft.fftfreq(self.padded_shape[0])[:, np.newaxis]
            u_x = np.fft.rfftfreq(self.padded_shape[1])[np.newaxis, :]
            return u_y, u_x
        elif len(self.padded_shape) == 1:
            u_x = np.fft.rfftfreq(self.padded_shape[0])
            return u_x
        else:
            raise ValueError('invalid self.shape: {}'.format(self.padded_shape))

    def u_abs(self):
        spatial_frequency_axes = self.u_axes()
        if len(spatial_frequency_axes) == 3:
            u_radial = spatial_frequency_axes[0] ** 2 + spatial_frequency_axes[1] ** 2 + spatial_frequency_axes[2] ** 2
        elif len(spatial_frequency_axes) == 2:
            u_radial = spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2
        else:
            u_radial = np.abs(spatial_frequency_axes)**2
        np.sqrt(u_radial, out=u_radial)
        return u_radial

    def u_abs2(self):
        spatial_frequency_axes = self.u_axes()
        if len(spatial_frequency_axes) == 3:
            u_radial = spatial_frequency_axes[0] ** 2 + spatial_frequency_axes[1] ** 2 + spatial_frequency_axes[2] ** 2
        elif len(spatial_frequency_axes) == 2:
            u_radial = spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2
        else:
            u_radial = np.abs(spatial_frequency_axes)**2
        return u_radial

    @property
    def u_radial(self):
        return self.u_abs()

    def __str__(self):
        return f'FourierFilter with shape {self.input_shape}, {self.padded_shape} (padded) {self.ft_shape} (ft), dtype {self.dtype}, pad length {self.pad_length}'


class PowerSpectrumCalculator(FourierFilterer):
    def __init__(self, imshape, pixel_size=1.0, zero_length=32):
        '''
        compute power spectra of n-dimensional images

        especially suited for 2D and 3D images

        windowing is done via a smoothed mask, which wastes little image area but has suboptimal precision properties

        precision at lower frequencies is bad due to the windowing used

        :param imshape:       shape of the image
        :param pixel_size:  sampling distance of the image
        :param zero_length: length from the outside in that is set to zero (smoothed)
        '''
        super().__init__(imshape, dtype='f8', pad_mode='constant', pad_constant_value=0.0)
        self.pixel_size = pixel_size
        self.zero_length = zero_length
        self._make_window()

    def _make_window(self):
        # a zero_length of 8 will produce a noticeable but small numerical error
        # a zero_length of 16 or larger produces no significant numerical errors (gaussian with std of >= 4 pixels)
        if self.zero_length < 16:
            print('warning: zero_length is too short')
        min_imshape = min(self.input_shape)
        if min_imshape < 2*self.zero_length:
            raise AssertionError('images must be larger then the zero area of the window')
        elif min_imshape < 64:
            self.zero_length = min_imshape // 4

        window_func = np.zeros(self.input_shape, dtype='f8')
        window_func[len(self.input_shape) * (slice(self.zero_length, -self.zero_length),)] = 1.

        gauss_blur_filter = MTF_gauss(self.u_abs(), self.zero_length / 4)
        self.window_func = self.irfftn(gauss_blur_filter*self.rfftn(window_func)).astype('f8')  # no padding required due to image contents being zero at the edges
        self.window_norm = self.pixel_size ** len(self.input_shape) / np.sum(self.window_func ** 2)

    def _pad(self, arr_in, for_power_spectrum=False):
        super()._pad(arr_in)

        if for_power_spectrum:
            self.arr[self.pad_slices] -= np.mean(self.arr[self.pad_slices])
            self.arr[self.pad_slices] *= self.window_func

    def power_spectrum(self, image: np.ndarray, shift=False):
        '''
        compute a power spectrum from an n-dimensional image

        :param image:   n-dimensional image
        :param shift:   use np.fft.fftshift on the result
        :return:        power spectrum shaped as the rfftn result
        '''

        fourier_transformed = self.rfftn(image, for_power_spectrum=True)
        if shift:
            fourier_transformed = np.fft.fftshift(fourier_transformed, axes=0)

        power_spectrum = fourier_transformed.real ** 2
        power_spectrum += fourier_transformed.imag ** 2
        power_spectrum *= self.window_norm
        return power_spectrum

    def auto_nbins(self):
        return int(25 + np.sqrt(max(self.u_radial.shape)))

    # radially averaged power spectra
    @staticmethod
    def average_fourier_function(frequency_coord, fourier_function, nbins='auto',
                                 exclude_average=True):
        ''' average a fourier space function over coordinates (e.g. radial frequency coords)

        :param frequency_coord:     radially symmetric spatial frequency coordinate of shape (N, M, ...)
        :param fourier_function:    fourier function (e.g. power spectrum) of shape (N, M, ...)
        :param nbins:               number of bins or 'auto' (default)
        :param exclude_average:     exclude the images average from the binning (True by default, strongly recommended)
        :param smooth:              smooth the spectrum
        :return:                    bin_edges (shape=nbins+1), radially_averaged_power_spectrum (shape=nbins),
                                    number of averaged values per bin (shape=nbins)
        '''
        if exclude_average:
            vrange = (1e-6, 0.5)
        else:
            vrange = (0, 0.5)
        if nbins == 'auto':
            nbins = int(25 + np.sqrt(max(frequency_coord.shape)))

        norm_hist, bin_edges = np.histogram(frequency_coord, nbins, range=vrange)
        hist, bin_edges = np.histogram(frequency_coord, nbins, range=vrange,
                                       weights=fourier_function)
        binned_spectrum = hist / np.fmax(norm_hist, 1)

        if norm_hist[0] == 0:
            bin_edges = bin_edges[1:]
            binned_spectrum = binned_spectrum[1:]
            norm_hist = norm_hist[1:]


        return bin_edges, binned_spectrum, norm_hist

    def radial_power_spectrum(self, image: np.ndarray, nbins='auto'):
        if nbins == 'auto':
            nbins = self.auto_nbins()

        PS = self.power_spectrum(image)
        bin_edges, radial_spectrum, norm_hist = self.average_fourier_function(self.u_radial, PS, nbins)

        return self.bin_edges_to_bin_centers(bin_edges), radial_spectrum

    def radial_power_spectrum_avg(self, images: np.ndarray, nbins:(str, int)='auto'):
        if nbins == 'auto':
            nbins = self.auto_nbins()

        for k, image in enumerate(images):
            if k == 0:
                PS_mean = self.power_spectrum(image)
            else:
                PS_mean += self.power_spectrum(image)
        PS_mean /= images.shape[0]

        bin_edges, radial_spectrum, norm_hist = self.average_fourier_function(self.u_radial, PS_mean, nbins)
        return self.bin_edges_to_bin_centers(bin_edges), radial_spectrum

    @staticmethod
    def bin_edges_to_bin_centers(bin_edges):
            bin_vals = np.zeros(len(bin_edges) - 1)
            for i in np.arange(len(bin_edges) - 1):
                bin_vals[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            return bin_vals


# padding functions (faster than numpy.set_data and can use output array)
def copy_range(arr, axis=0, range_r=(None, None), range_w=(None, None), invert_read=False):
    all_slice = slice(None, None)
    slices_r = tuple(all_slice if k != axis else slice(range_r[0], range_r[1]) for k in range(arr.ndim))
    slices_w = tuple(all_slice if k != axis else slice(range_w[0], range_w[1]) for k in range(arr.ndim))
    #print(slices_r, slices_w)

    if invert_read:
        invert_slice = tuple(all_slice if k != axis else slice(None, None, -1) for k in range(arr.ndim))
        arr[slices_w] = arr[slices_r][invert_slice]
    else:
        arr[slices_w] = arr[slices_r]


def get_fill_slice(pad_lengths):
    slices = []
    for pad_range in pad_lengths:
        slices.append(slice(pad_range[0], -pad_range[1] if pad_range[1] > 0 else None))
    return tuple(slices)


def pad_reflect(arr_in, arr_out, pad_lengths):
    arr_out[get_fill_slice(pad_lengths)] = arr_in

    for axis in range(arr_in.ndim):
        if pad_lengths[axis][0] > 0:
            pad_before = min(pad_lengths[axis][0], arr_in.shape[axis]-1)
            rest_before = pad_lengths[axis][0] - pad_before
            copy_range(arr_out, axis, (pad_lengths[axis][0]+1, pad_lengths[axis][0]+pad_before+1),
                       (rest_before, pad_before+rest_before), True)

        if pad_lengths[axis][1] > 0:
            pad_after = min(pad_lengths[axis][1], arr_in.shape[axis]-1)
            rest_after = pad_lengths[axis][1] - pad_after
            copy_range(arr_out, axis, (-pad_after-pad_lengths[axis][1]-1, -pad_lengths[axis][1]-1),
                        (-pad_after-rest_after, None if rest_after == 0 else -rest_after), True)


def pad_constant(arr_in, arr_out, pad_lengths, val=0.):
    arr_out[:] = val
    arr_out[get_fill_slice(pad_lengths)] = arr_in
    
    
# ==== Fourier filter functions ====
def MTF_exp(coords, mu):
    return np.exp(-coords*mu*2*np.pi)


def MTF_gauss(coords, sigma):
    return np.exp(-2*np.pi**2*coords**2*sigma**2*np.sign(sigma))


def MTF_voigt(coords, sigma, mu):
    return np.exp(-2*np.pi**2*coords**2*sigma**2*np.sign(sigma)-coords*mu*2*np.pi)


def MTF_sinc(coords, width):
    return np.sinc(coords*width)


# ==== choosing padded lengths for fast FFT ====
def get_fast_pad_lengths(data_length, pad_length=None, pad_lengths=None):
    if pad_length is not None:
        padded_length = data_length + 2 * pad_length
    elif pad_lengths is not None:
        padded_length = data_length + pad_lengths[0] + pad_lengths[1]
    else:
        padded_length = data_length

    padded_length = np.amin(get_fast_fft_sizes(padded_length, 2 * padded_length))

    if pad_lengths is None:
        pad_diff = padded_length - data_length
        pad_before = pad_diff // 2
        pad_after = pad_diff - pad_before
    else:
        pad_diff = padded_length - data_length - pad_lengths[0] - pad_lengths[1]
        pad_before = pad_lengths[0] + pad_diff // 2
        pad_after = pad_diff - pad_before
    # print('set_data lengths', pad_before, pad_after, data_length, pad_lengths, pad_length)
    return pad_before, pad_after


def get_fast_fft_sizes(min_length, max_length=4096, other_radices=(3, 5), min_2pot=3):
    radix_conributions = []
    radix_conributions.append(2 ** np.arange(min_2pot, int(np.log(max_length) / np.log(2)) + 1))
    for k, radix in enumerate(other_radices):
        radix_conributions.append(radix ** np.arange(0, int(np.log(max_length / 2 ** min_2pot) / np.log(radix)) + 1))

    n = 1 + len(other_radices)
    all_sizes = np.prod([radix_conributions[k][nd_newaxis_other(k, n)] for k in np.arange(n)]).astype('i8')
    all_sizes = all_sizes[np.logical_and(np.greater_equal(all_sizes, min_length), np.less_equal(all_sizes, max_length))]
    all_sizes.sort()

    return all_sizes


def nd_newaxis_other(k, ndim):
    return tuple(slice(None) if m == k else None for m in range(ndim))


PRNE = 3  # pad_range_norm_exponent, global constant! higher values lead to faster filtering, if filters are combined
          # the filter errors for PRNE < 1 are in the area that is then used for padding itself
          # => they should not matter much for well-behaved filters


def sum_pad_ranges(ranges_list):  # add function for set_data ranges
    return int(np.ceil(sum(np.array(ranges_list) ** PRNE) ** (1 / PRNE)))


def abs_u(spatial_frequency_axes):
        if len(spatial_frequency_axes) == 3:
            return np.sqrt(spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2 + spatial_frequency_axes[2]**2)
        elif len(spatial_frequency_axes) == 2:
            return np.sqrt(spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2)
        elif len(spatial_frequency_axes) == 1:
            return np.abs(spatial_frequency_axes[0])


def abs_u2(spatial_frequency_axes):
        if len(spatial_frequency_axes) == 3:
            return spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2 + spatial_frequency_axes[2]**2
        elif len(spatial_frequency_axes) == 2:
            return spatial_frequency_axes[0]**2 + spatial_frequency_axes[1]**2
        elif len(spatial_frequency_axes) == 1:
            return spatial_frequency_axes[0]**2


# analytic wiener filter (robust)
def _get_wiener_filter_analytic(fourier_transform, deconv_filter, noise_sup_strength, alpha=1.):
    scaled_spatial_freq = fourier_transform.spatial_freq()/fourier_transform.u_ny
    deconv_filter = (1 + 1e-20) / (1/deconv_filter + 1e-20)  # regularization for purely numerical reasons
    NSR = (scaled_spatial_freq**alpha*noise_sup_strength*deconv_filter)**2 # assumes the signal to be roughly proportional to u^-1
    return 1/(1 + NSR)


def _get_wiener_deconv_filter_analytic(fourier_transform, deconv_mu=0., deconv_std=0., noise_sup_strength=0.05, reg=0.02, alpha=1.):
    scaled_spatial_freq = fourier_transform.spatial_freq()/fourier_transform.u_ny
    deconv_filter = np.ones_like(scaled_spatial_freq)
    if not np.isclose(deconv_std, 0.):
        deconv_filter *= MTF_gauss(fourier_transform, -deconv_std)
    if not np.isclose(deconv_mu, 0.):
        deconv_filter *= MTF_exp(fourier_transform, -deconv_mu)
    wiener_filter = _get_wiener_filter_analytic(fourier_transform, deconv_filter, noise_sup_strength, alpha=alpha)
    return deconv_filter*wiener_filter


def _get_wiener_filter_analytic_with_spectrum(fourier_transform, deconv_filter, spectrum, preserve_average=True):
    scaled_spatial_freq = fourier_transform.spatial_freq()/fourier_transform.u_ny
    deconv_filter = (1 + 1e-20) / (1/deconv_filter + 1e-20)  # regularization for purely numerical reasons
    NSR = (spectrum*deconv_filter)**2 # assumes the signal to be roughly proportional to u^-1
    if preserve_average:
        if scaled_spatial_freq.ndims == 1:      NSR[0] = 0.
        elif scaled_spatial_freq.ndims == 2:    NSR[0, 0] = 0.
        elif scaled_spatial_freq.ndims == 3:    NSR[0, 0, 0] = 0.
    return 1/(1 + NSR)


def _wiener_deconv_filter_analytic_slow(u, u_ny, sigma, mu, noise_sup_strength, reg=0.02, alpha=1., phase_parameter=0.):
    deconv_filter = 1/(exp(-u*mu*2*pi)*exp(-2*pi**2*u**2*sigma**2*sign(sigma)))
    # sign(sigma) converts negative values to a smoothing kernel
    deconv_filter = (1 + 1e-20) / (1/deconv_filter + 1e-20)

    NSR = ((u/u_ny)**alpha*noise_sup_strength*deconv_filter/(1+phase_parameter**2*u**2))**2 # assumes the signal to be roughly proportional to u^-alpha
    #print('mean(NSR)', mean(NSR))
    wiener_filter = 1/(1 + NSR)

    deconv_filter = (1 + reg) / (1/deconv_filter + reg)
    return deconv_filter*wiener_filter


def wiener_deconv_filter_analytic(u, u_ny, sigma, mu, noise_sup_strength, reg=0.02, alpha=1., phase_parameter=0.):

    if not np.isclose(mu, 0.):
        exp_arg = np.copy(u)
        exp_arg *= mu*2*np.pi
    else:
        exp_arg = 0.
    if not np.isclose(sigma, 0.):
        gauss_arg = u**2
        gauss_arg *= 2*np.pi**2*sigma**2*np.sign(sigma) # sign(sigma) converts negative values to a smoothing kernel
    else:
        gauss_arg = 0.

    if not np.isclose(mu, 0.) or not np.isclose(sigma, 0.) :
        deconv_filter = np.exp(exp_arg + gauss_arg)
        np.fmax(deconv_filter, reg, out=deconv_filter)
        del exp_arg, gauss_arg
    else:
        deconv_filter = 1.

    if np.isclose(alpha, 1.):
        noise_sup_strength = noise_sup_strength/u_ny
        if not np.isclose(phase_parameter, 0.):
            NSR = u*noise_sup_strength/(1+phase_parameter**2*u**2) # assumes the signal to be roughly proportional to u^-1
        else:
            NSR = u*noise_sup_strength # assumes the signal to be roughly proportional to u^-1
    else:
        if not np.isclose(phase_parameter, 0.):
            NSR = (u/u_ny)**alpha*noise_sup_strength/(1+phase_parameter**2*u**2) # assumes the signal to be roughly proportional to u^-alpha
        else:
            NSR = (u/u_ny)**alpha*noise_sup_strength # assumes the signal to be roughly proportional to u^-alpha
    NSR *= deconv_filter
    NSR **= 2
    #print('mean(NSR)', mean(NSR))
    NSR += 1 # instead of wiener_filter = 1(1+NSR)
    NSR **= -1
    wiener_filter = NSR
    wiener_filter *= deconv_filter
    return wiener_filter
