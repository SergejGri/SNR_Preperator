''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import numpy as np
import ext.common.opencl as cl
from ext.image.resize import bin_down_image

# ==== image registration in OpenCl ====
class ImageRegisterTool():
    _precision_steps = (1/3)**np.arange(0, 6)
    block_size = np.int32(64)  # best performance for 32 to 128
    local_size_3d = (64, 2, 2)  # performance is almost completely independent of this
    local_size_2d = (16, 16)
    verbose = 1  # 0 for disabled, 1 for partial (recommended), 2 for full (+intermediate) printout of results

    def __init__(self, compute_device, bin_factor=2, min_iter_search_steps=15, highpass_limit=None):
        '''


        :param bin_factor:  bin factor for a fast wide search in the first step, large values are very fast
                            but may result in the registration failing
        :param min_iter_search_steps:   minimum steps to search at the precision search where step size is 1/3 of the step before
                                        the value used is always >= 5 (see iter_search_steps())
        :return:
        '''
        self.compute_device = compute_device
        self.bin_factor = int(np.clip(bin_factor, 1, 9))
        self._iter_search_steps = min_iter_search_steps

        self.highpass_limit = highpass_limit
        self.fourier_filter = None

        self.cl_program = cl.Program(self.compute_device, self.cl_source)

        self.image_shape = [0, 0]
        self.image_1_cl = [None, None]
        self.image_2_cl = [None, None]
        self.diff_norm = [None, None]
        self.mask_shape = [0, 0]
        self.mask1_cl = [None, None]
        self.mask2_cl = [None, None]
        self.use_mask = False
        self.has_mask_cl_functions = False

        self.result_shape = [0, 0]
        self.result_values = [None, None]
        self.result_weights = [None, None]
        self.result_shape_cl = [None, None]
        self.result_values_cl = [None, None]
        self.result_weights_cl = [None, None]

        self.diff_images = [None, None, None, None]
        self.diff_coords = [None, None, None, None]

    # common functions
    def register_stack(self, images_1, images_2, xrange, yrange, method='shift',
                       progress_func=lambda n, N: None, **kwargs):
        if method == 'shift':
            length = images_1.shape[0]
            xshifts = np.zeros(length)
            yshifts = np.zeros(length)
            diff_measure_images_param = []

            for k in np.arange(length):
                xshifts[k], yshifts[k], opt_indices, diff_measure_image = self.auto_find_shift(
                    images_1[k], images_2[k], yrange, xrange, **kwargs)
                diff_measure_images_param.append(diff_measure_image)
                progress_func(k + 1, length)

            return xshifts, yshifts, diff_measure_images_param

        elif method == 'shiftscale':
            length = images_1.shape[0]
            xshifts = np.zeros(length)
            yshifts = np.zeros(length)
            scales = np.zeros(length)
            diff_measure_images_param = []

            for k in np.arange(length):
                xshifts[k], yshifts[k], scales[k], opt_indices, diff_measure_images = self.auto_find_shift_scale(
                    images_1[k], images_2[k], yrange, xrange, **kwargs)
                diff_measure_images_param.append(diff_measure_images[opt_indices[0]])

            return xshifts, yshifts, scales, diff_measure_images_param

        elif method == 'shiftrot':
            length = images_1.shape[0]
            xshifts = np.zeros(length)
            yshifts = np.zeros(length)
            rot_angles = np.zeros(length)
            diff_measure_images_param = []

            for k in np.arange(length):
                xshifts[k], yshifts[k], rot_angles[k], opt_indices, diff_measure_images = self.auto_find_shift_rot(
                    images_1[k], images_2[k], yrange, xrange, **kwargs)
                diff_measure_images_param.append(diff_measure_images[opt_indices[0]])

            return xshifts, yshifts, rot_angles, diff_measure_images_param

        else:
            raise ValueError('method must be one of "shift", "shiftscale", "shiftrot"')

    def remove_nonfinite(self, image):
        np.copyto(image, 0., where=~np.isfinite(image))

    def allocate_images_mem(self, image_1, image_2):
        image_1 = image_1.astype("f4", copy=False)
        image_2 = image_2.astype("f4", copy=False)
        self.remove_nonfinite(image_1)
        self.remove_nonfinite(image_2)
        if self.highpass_limit is not None:
            image_1 = self.filter_highpass(image_1)
            image_2 = self.filter_highpass(image_2)
            if self.verbose == 2: print('applied highpass')

        for use_binned in (0, 1):
            if use_binned > 0:
                image_1 = bin_down_image(image_1, self.bin_factor)
                image_2 = bin_down_image(image_2, self.bin_factor)

            image_shape = np.array(image_1.shape)
            if not np.all(image_shape == self.image_shape[use_binned]):
                self.image_shape[use_binned] = image_shape
                self.image_1_cl[use_binned] = cl.Image(self.compute_device,
                                                           cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                                           arr=image_1)
                self.image_2_cl[use_binned] = cl.Image(self.compute_device,
                                                           cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                                           arr=image_2)
            else:
                self.image_1_cl[use_binned].copy_from(image_1)
                self.image_2_cl[use_binned].copy_from(image_2)

            self.diff_norm[use_binned] = np.std(image_1) * np.std(image_2) / (
                    cl.cl_float_unscaler(image_1.dtype) * cl.cl_float_unscaler(image_2.dtype))

    def allocate_result_mem(self, nparams):
        for use_binned in (0, 1):
            result_shape = np.array((nparams, *np.ceil(np.array(self.image_shape[use_binned]) / self.block_size)), 'i4')
            if not np.all(result_shape == self.result_shape[use_binned]):
                self.result_shape[use_binned] = result_shape
                self.result_values[use_binned] = np.zeros(result_shape, 'f4')
                self.result_weights[use_binned] = np.zeros(result_shape, 'f4')
                self.result_shape_cl[use_binned] = np.array((result_shape[2], result_shape[1], result_shape[0], 0), 'i4')

                self.result_values_cl[use_binned] = cl.Buffer(self.compute_device,
                                                                  cl.Buffer.MemFlags.WRITE_ONLY | cl.Buffer.MemFlags.HOST_READ_ONLY,
                                                                  size=self.result_values[use_binned].nbytes)
                self.result_weights_cl[use_binned] = cl.Buffer(self.compute_device,
                                                                   cl.Buffer.MemFlags.WRITE_ONLY | cl.Buffer.MemFlags.HOST_READ_ONLY,
                                                                   size=self.result_weights[use_binned].nbytes)

    def set_mask(self, mask1: np.ndarray, mask2: np.ndarray =None):
        self.use_mask = mask1 is not None

        if mask1 is not None:
            if mask1.mean() < 0.5 or (mask2 is not None and mask2.mean() < 0.5):
                print('WARNING: mask is mostly empty, inverted mask?')
            mask1 = (mask1 > 0).astype('f4' ) *255
            if mask2 is not None:
                mask2 = (mask2 > 0).astype('f4' ) *255
            for use_binned in (0, 1):
                if use_binned > 0:
                    mask1 = bin_down_image(mask1.astype('f4', copy=False), self.bin_factor)
                self.mask_shape[use_binned] = mask1.shape
                mask1 = mask1.astype('u1')
                self.mask1_cl[use_binned] = cl.Image(self.compute_device,
                                                         cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                                         arr=mask1)

                if mask2 is not None:
                    if use_binned > 0:
                        mask2 = bin_down_image(mask2.astype('f4', copy=False), self.bin_factor)
                    if mask2 is not None: mask2 = bin_down_image(mask2, self.bin_factor)
                    assert np.all(np.array(mask2.shape) == self.image_shape[use_binned])
                    mask2 = mask2.astype('u1')
                    self.mask2_cl[use_binned] = cl.Image(self.compute_device,
                                                             cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                                             arr=mask2)
                else:
                    self.mask2_cl[use_binned] = self.mask1_cl[use_binned]

                # print("mask image format", cl_valid_float_format(mask1.dtype), "mask\n", mask1)
            print('mask set, mean: {:.4f}'.format(mask1.mean( ) /255))

            if not self.has_mask_cl_functions:
                self.cl_program_mask = cl.Program(self.compute_device, self.cl_source_masked)
                self.has_mask_cl_functions = True

    def precision_steps(self, grade=2):
        return (self.bin_factor, *self._precision_steps[self.bin_factor < 2:grade+1])

    def iter_search_steps(self, precision_factor, used_binned):
        return int(np.ceil(2 * (1 + used_binned) * precision_factor))

    # shift registration
    def auto_find_shift(self, image_1, image_2, xrange, yrange, precision_grade=2):
        last_precision = None

        for k, precision in enumerate(self.precision_steps(precision_grade)):
            if k == 0:
                yshifts = np.unique(np.hstack((np.arange(yrange[0], yrange[1], precision), yrange[1])))
                xshifts = np.unique(np.hstack((np.arange(xrange[0], xrange[1], precision), xrange[1])))
            else:
                search_steps = self.iter_search_steps(last_precision / precision, k == 1)
                yshifts = np.linspace(yshift_opt - search_steps * precision, yshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)
                xshifts = np.linspace(xshift_opt - search_steps * precision, xshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)

            if self.verbose == 2:
                print('searching xshift [{:.2f} {:.2f} {:.2f}], yshift  [{:.2f} {:.2f} {:.2f}]'.format(
                    xshifts.min(), xshifts.max(), precision, yshifts.min(), yshifts.max(), precision))
            elif self.verbose == 1 and k == 0:
                print('searching xshift [{:.2f} {:.2f}], yshift [{:.2f} {:.2f}]]'.format(
                    xshifts.min(), xshifts.max(), yshifts.min(), yshifts.max()))

            xshift_opt, yshift_opt, opt_indices, diff_measure_image_ = self.find_shift(
                image_1, image_2, xshifts, yshifts, use_binned=k == 0)

            if self.verbose == 2 or (self.verbose == 1 and k == len(self.precision_steps(precision_grade)) - 1):
                print('found shift ({:.2f}, {:.2f}), normed diff min {:.4f}'.format(xshift_opt, yshift_opt,
                                                                                    diff_measure_image_.min()))
            if (self.bin_factor > 1 and k == 1) or (k == 0):
                diff_measure_image = diff_measure_image_
            if self.verbose == 2:
                self.diff_images[k] = diff_measure_image_
                self.diff_coords[k] = xshifts, yshifts
            last_precision = precision

        if np.any(np.isclose(yshift_opt, yrange)) or np.any(np.isclose(xshift_opt, xrange)):
            print('WARNING: optimal shift values not found')

        return xshift_opt, yshift_opt, opt_indices[:2], diff_measure_image

    def find_shift(self, image_1, image_2, xshifts, yshifts, allocate=True, use_binned=0):
        diff_measure = np.zeros((len(yshifts), len(xshifts)), 'f4')

        if allocate:
            self.allocate_images_mem(image_1, image_2)

        yshifts = yshifts[np.logical_and(-self.image_shape[use_binned][0] + 20 < yshifts,
                                      yshifts < self.image_shape[use_binned][0] - 20)]
        xshifts = xshifts[np.logical_and(-self.image_shape[use_binned][1] + 20 < xshifts,
                                      xshifts < self.image_shape[use_binned][1] - 20)]
        nparams = len(yshifts) * len(xshifts)
        if allocate:
            self.allocate_result_mem(nparams)

        shift_vecs = np.zeros((nparams, 2), 'f4')
        x_size = len(xshifts)
        for yindex, yshift in enumerate(yshifts):
            zy_offset = yindex * x_size
            shift_vecs[zy_offset:zy_offset + x_size, 1] = yshift
            for xindex, xshift in enumerate(xshifts):
                shift_vecs[zy_offset + xindex, 0] = xshift
                # todo: write faster code

        if use_binned:
            shift_vecs /= self.bin_factor ** use_binned
        shift_vecs_cl = cl.Buffer(self.compute_device,
                                      cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                      arr=shift_vecs)

        if np.allclose((xshifts[1:] - xshifts[:-1]) % 1, 0.) and np.allclose((yshifts[1:] - yshifts[:-1]) % 1, 0.):
            if not self.use_mask:
                self.cl_program.shifted_params_square_diff_2d(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                              self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                              self.result_values_cl[use_binned],
                                                              self.result_weights_cl[use_binned],
                                                              self.result_shape_cl[use_binned], self.block_size,
                                                              shift_vecs_cl)
            else:
                self.cl_program_mask.shifted_params_square_diff_2d(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                                   self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                                   self.mask1_cl[use_binned], self.mask2_cl[use_binned],
                                                                   self.result_values_cl[use_binned],
                                                                   self.result_weights_cl[use_binned],
                                                                   self.result_shape_cl[use_binned], self.block_size,
                                                                   shift_vecs_cl)
            # print('using int shift kernel')
        else:
            if not self.use_mask:
                self.cl_program.shifted_params_square_diff_2d_subpixel(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                                       self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                                       self.result_values_cl[use_binned],
                                                                       self.result_weights_cl[use_binned],
                                                                       self.result_shape_cl[use_binned], self.block_size,
                                                                       shift_vecs_cl)
            else:
                self.cl_program_mask.shifted_params_square_diff_2d_subpixel(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                                            self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                                            self.mask1_cl[use_binned], self.mask2_cl[use_binned],
                                                                            self.result_values_cl[use_binned],
                                                                            self.result_weights_cl[use_binned],
                                                                            self.result_shape_cl[use_binned], self.block_size,
                                                                            shift_vecs_cl)
            # print('using subpixel kernel')

        self.result_values_cl[use_binned].copy_to(self.result_values[use_binned])
        self.result_weights_cl[use_binned].copy_to(self.result_weights[use_binned])



        diff_measure[:, :].flat = np.sum(self.result_values[use_binned], axis=(1, 2)) / np.sum(
            self.result_weights[use_binned], axis=(1, 2))
        diff_measure[np.isnan(diff_measure)] = np.inf
        diff_measure /= self.diff_norm[use_binned]

        opt_indices = np.unravel_index(np.argmin(diff_measure), diff_measure.shape)
        xshift_opt, yshift_opt = xshifts[opt_indices[1]], yshifts[opt_indices[0]]

        diff_measure_image = diff_measure.reshape(diff_measure.shape)

        return xshift_opt, yshift_opt, opt_indices, diff_measure_image

    def shift_image(self, image, xshift=0., yshift=0., invert_transform=False):
        image_out = np.zeros_like(image, 'f4')
        image_in_cl = cl.Image(self.compute_device,
                                   cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                   arr=image.astype('f4'))
        image_out_cl = cl.Image(self.compute_device,
                                    cl.Buffer.MemFlags.WRITE_ONLY | cl.Buffer.MemFlags.HOST_READ_ONLY,
                                    cl.im_format_float, image.shape)
        shift_vec = np.array((xshift, yshift), 'f4')
        if invert_transform:
            shift_vec *= -1

        self.cl_program.shift_image(image.shape[::-1], self.local_size_2d,
                                    image_in_cl, image_out_cl, shift_vec)
        image_out_cl.copy_to(image_out)
        return image_out

    # shift+scale registration
    def find_shift_scale(self, image_1, image_2, xshifts, yshifts, scale_factors=(1.0,), scale_origin=None,
                         allocate=True, use_binned=0):

        if allocate:
            self.allocate_images_mem(image_1, image_2)
        yshifts = yshifts[np.logical_and(-self.image_shape[use_binned][0] + 20 < yshifts,
                                      yshifts < self.image_shape[use_binned][0] - 20)]
        xshifts = xshifts[np.logical_and(-self.image_shape[use_binned][1] + 20 < xshifts,
                                      xshifts < self.image_shape[use_binned][1] - 20)]
        nparams = len(yshifts) * len(xshifts)
        if allocate:
            self.allocate_result_mem(nparams)

        shift_vecs = np.zeros((nparams, 2), 'f4')
        x_size = len(xshifts)
        for yindex, yshift in enumerate(yshifts):
            zy_offset = yindex * x_size
            shift_vecs[zy_offset:zy_offset + x_size, 1] = yshift
            for xindex, xshift in enumerate(xshifts):
                shift_vecs[zy_offset + xindex, 0] = xshift
                # todo: write faster code
        if scale_origin is None:
            scale_origin = (np.array(image_1.shape, 'f4')[::-1] / 2)
        else:
            scale_origin = np.array(scale_origin, 'f4')
        if use_binned:
            shift_vecs /= self.bin_factor ** use_binned
            scale_origin /= self.bin_factor ** use_binned
        shift_vecs_cl = cl.Buffer(self.compute_device,
                                      cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                      arr=shift_vecs)

        diff_measure_images = np.zeros((len(scale_factors), len(yshifts), len(xshifts)), 'f4')
        for zindex, scale_factor in enumerate(scale_factors):
            scale_factor = np.float32(scale_factor)

            self.cl_program.shiftscaled_params_square_diff_2d(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                              self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                              self.result_values_cl[use_binned],
                                                              self.result_weights_cl[use_binned],
                                                              self.result_shape_cl[use_binned], self.block_size,
                                                              shift_vecs_cl, scale_origin, scale_factor)

            self.result_values_cl[use_binned].copy_to(self.result_values[use_binned])
            self.result_weights_cl[use_binned].copy_to(self.result_weights[use_binned])
            diff_measure_images[zindex, :, :].flat = np.sum(self.result_values[use_binned], axis=(1, 2)) / np.sum(
                self.result_weights[use_binned], axis=(1, 2))

        diff_measure_images[np.isnan(diff_measure_images)] = np.inf
        diff_measure_images /= self.diff_norm[use_binned]

        opt_indices = np.unravel_index(np.argmin(diff_measure_images), diff_measure_images.shape)
        xshift_opt, yshift_opt, scale_fact_opt = xshifts[opt_indices[2]], yshifts[opt_indices[1]], scale_factors[
            opt_indices[0]]

        return xshift_opt, yshift_opt, scale_fact_opt, opt_indices, diff_measure_images

    def auto_find_shift_scale(self, image_1, image_2, yrange, xrange, scale_range, scale_origin=None,
                              start_scale_step=0.01, precision_grade=2):
        last_precision = None

        for k, precision in enumerate(self.precision_steps(precision_grade)):
            if k == 0:
                yshifts = np.unique(np.hstack((np.arange(yrange[0], yrange[1], precision), yrange[1])))
                xshifts = np.unique(np.hstack((np.arange(xrange[0], xrange[1], precision), xrange[1])))
                scale_factors = np.linspace(*scale_range, (scale_range[1] - scale_range[0]) // start_scale_step + 2)
            else:
                search_steps = self.iter_search_steps(last_precision / precision, k == 1)
                yshifts = np.linspace(yshift_opt - search_steps * precision, yshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)
                xshifts = np.linspace(xshift_opt - search_steps * precision, xshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)
                if k > 1 or self.bin_factor == 1:
                    scale_factors = scale_fact_opt + np.linspace(-1.5 * scale_step, 1.5 * scale_step, 17)
            scale_step = scale_factors[1] - scale_factors[0]

            if self.verbose == 2:
                print(
                    'searching xshift [{:.2f} {:.2f} {:.2f}], yshift [{:.2f} {:.2f} {:.2f}], scale [{:.4f} {:.4f} {:.5f}]'.format(
                        xshifts.min(), xshifts.max(), precision, yshifts.min(), yshifts.max(),
                        precision, scale_factors.min(), scale_factors.max(), scale_step))
            elif self.verbose == 1 and k == 0:
                print('searching xshift [{:.2f} {:.2f}], yshift [{:.2f} {:.2f}], scale [{:.2f} {:.2f}]'.format(
                    xshifts.min(), xshifts.max(), yshifts.min(), yshifts.max(), scale_factors.min(),
                    scale_factors.max()))

            xshift_opt, yshift_opt, scale_fact_opt, opt_indices, diff_measure_images_ = self.find_shift_scale(
                image_1, image_2, xshifts, yshifts, scale_factors, scale_origin, use_binned=k == 0)
            if self.verbose == 2 or (self.verbose == 1 and k == len(self.precision_steps(precision_grade)) - 1):
                print('found shift ({:.2f}, {:.2f}), scale {:.4f}, mindiff {:.3f}'.format(xshift_opt,
                                                                                          yshift_opt, scale_fact_opt,
                                                                                          diff_measure_images_.min()))
            if (self.bin_factor > 1 and k == 1) or (k == 0):
                diff_measure_images = diff_measure_images_
            last_precision = precision

        if np.any(np.isclose(yshift_opt, yrange)) or np.any(np.isclose(xshift_opt, xrange)) or np.any(
                np.isclose(scale_fact_opt, scale_range)):
            print('WARNING: optimal shift/scale values not found')

        return xshift_opt, yshift_opt, scale_fact_opt, opt_indices, diff_measure_images

    def shift_scale_image(self, image, xshift=0., yshift=0., scale_factor=1., scale_origin=None,
                          invert_transform=False):

        image_out = np.zeros_like(image, 'f4')
        image_in_cl = cl.Image(self.compute_device,
                                   cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                   arr=image.astype('f4'))
        image_out_cl = cl.Image(self.compute_device,
                                    cl.Buffer.MemFlags.WRITE_ONLY | cl.Buffer.MemFlags.HOST_READ_ONLY,
                                    cl.im_format_float, image.shape)
        scale_factor = np.float32(scale_factor)
        shift_vec = np.array((xshift, yshift), 'f4')
        if scale_origin is None:
            scale_origin = (np.array(image.shape, 'f4')[::-1] / 2)
        else:
            scale_origin = np.array(scale_origin, 'f4')

        print(shift_vec, scale_factor, type(shift_vec), type(scale_factor))

        if invert_transform:
            self.cl_program.shiftscale_image_inv(image.shape[::-1], self.local_size_2d,
                                                 image_in_cl, image_out_cl, shift_vec, scale_origin, scale_factor)

        else:
            self.cl_program.shiftscale_image(image.shape[::-1], self.local_size_2d,
                                             image_in_cl, image_out_cl, shift_vec, scale_origin, scale_factor)
        image_out_cl.copy_to(image_out)
        return image_out

    # shift+rotation registration
    def find_shift_rot(self, image_1, image_2, xshifts, yshifts, rot_angles=(1.0,), rot_origin=None, allocate=True,
                       use_binned=0):

        if allocate:
            self.allocate_images_mem(image_1, image_2)
        yshifts = yshifts[np.logical_and(-self.image_shape[use_binned][0] + 20 < yshifts,
                                      yshifts < self.image_shape[use_binned][0] - 20)]
        xshifts = xshifts[np.logical_and(-self.image_shape[use_binned][1] + 20 < xshifts,
                                      xshifts < self.image_shape[use_binned][1] - 20)]
        nparams = len(yshifts) * len(xshifts)
        if allocate:
            self.allocate_result_mem(nparams)

        shift_vecs = np.zeros((nparams, 2), 'f4')
        x_size = len(xshifts)
        for yindex, yshift in enumerate(yshifts):
            zy_offset = yindex * x_size
            shift_vecs[zy_offset:zy_offset + x_size, 1] = yshift
            for xindex, xshift in enumerate(xshifts):
                shift_vecs[zy_offset + xindex, 0] = xshift
                # todo: write faster code
        if rot_origin is None:
            rot_origin = (np.array(image_1.shape, 'f4')[::-1] / 2)
        else:
            rot_origin = np.array(rot_origin, 'f4')
        if use_binned:
            shift_vecs /= self.bin_factor ** use_binned
            rot_origin /= self.bin_factor ** use_binned
        shift_vecs_cl = cl.Buffer(self.compute_device,
                                      cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                      arr=shift_vecs)

        diff_measure_images = np.zeros((len(rot_angles), len(yshifts), len(xshifts)), 'f4')
        for zindex, rot_angle in enumerate(rot_angles):
            # if np.isclose(rot_angle, 0.) and (not np.isclose((xshifts[1]-xshifts[0])%1, 0.0) or not np.isclose((yshifts[1]-yshifts[0])%1, 0.0)):
            #    print('WARNING: subsampling produces false results')
            rot_angle = np.float32(np.deg2rad(rot_angle))
            self.cl_program.shiftrot_params_square_diff_2d(self.result_shape[use_binned][::-1], self.local_size_3d,
                                                           self.image_1_cl[use_binned], self.image_2_cl[use_binned],
                                                           self.result_values_cl[use_binned],
                                                           self.result_weights_cl[use_binned],
                                                           self.result_shape_cl[use_binned], self.block_size,
                                                           shift_vecs_cl, rot_origin, rot_angle)

            self.result_values_cl[use_binned].copy_to(self.result_values[use_binned])
            self.result_weights_cl[use_binned].copy_to(self.result_weights[use_binned])
            diff_measure_images[zindex, :, :].flat = np.sum(self.result_values[use_binned], axis=(1, 2)) / np.sum(
                self.result_weights[use_binned], axis=(1, 2))

        diff_measure_images /= self.diff_norm[use_binned]

        opt_indices = np.unravel_index(np.argmin(diff_measure_images), diff_measure_images.shape)
        xshift_opt, yshift_opt, rot_angle_opt = xshifts[opt_indices[2]], yshifts[opt_indices[1]], rot_angles[opt_indices[0]]

        return xshift_opt, yshift_opt, rot_angle_opt, opt_indices, diff_measure_images

    def auto_find_shift_rot(self, image_1, image_2, xrange, yrange, rot_angle_range, rot_origin=None,
                            start_angle_step=0.2, precision_grade=2):
        last_precision = None

        for k, precision in enumerate(self.precision_steps(precision_grade)):
            if k == 0:
                yshifts = np.unique(np.hstack((np.arange(yrange[0], yrange[1], precision), yrange[1])))
                xshifts = np.unique(np.hstack((np.arange(xrange[0], xrange[1], precision), xrange[1])))
                rot_angles = np.linspace(*rot_angle_range,
                                      (rot_angle_range[1] - rot_angle_range[0]) // start_angle_step + 2)
            else:
                search_steps = self.iter_search_steps(last_precision / precision, k == 1)
                yshifts = np.linspace(yshift_opt - search_steps * precision, yshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)
                xshifts = np.linspace(xshift_opt - search_steps * precision, xshift_opt + search_steps * precision,
                                   search_steps * 2 + 1, endpoint=True)
                if k > 1 or self.bin_factor == 1:
                    rot_angles = rot_angle_opt + np.linspace(-1.5 * angle_step, 1.5 * angle_step, 13)
            angle_step = rot_angles[1] - rot_angles[0]

            if self.verbose == 2:
                print(
                    'searching xshift [{:.2f} {:.2f} {:.2f}], yshift [{:.2f} {:.2f} {:.2f}], rot angle [{:.4f} {:.4f} {:.5f}]'.format(
                        xshifts.min(), xshifts.max(), precision, yshifts.min(), yshifts.max(),
                        precision, rot_angles.min(), rot_angles.max(), angle_step))
            elif self.verbose == 1 and k == 0:
                print('searching xshift [{:.2f} {:.2f}], yshift [{:.2f} {:.2f}], rot angle [{:.2f} {:.2f}]'.format(
                    xshifts.min(), xshifts.max(), yshifts.min(), yshifts.max(), rot_angles.min(), rot_angles.max()))

            xshift_opt, yshift_opt, rot_angle_opt, opt_indices, diff_measure_images_ = self.find_shift_rot(
                image_1, image_2, xshifts, yshifts, rot_angles, rot_origin, use_binned=k == 0)

            if self.verbose == 2 or (self.verbose == 1 and k == len(self.precision_steps(precision_grade)) - 1):
                print('found shift ({:.2f}, {:.2f}), rot angle {:.4f}, mindiff {:.3f}'.format(xshift_opt,
                                                                                              yshift_opt, rot_angle_opt,
                                                                                              diff_measure_images_.min()))
            if (self.bin_factor > 1 and k == 1) or (k == 0):
                diff_measure_images = diff_measure_images_
            last_precision = precision

        if np.any(np.isclose(yshift_opt, yrange)) or np.any(np.isclose(xshift_opt, xrange)) or np.any(
                np.isclose(rot_angle_opt, rot_angle_range)):
            print('WARNING: optimal shift/scale values not found')

        return xshift_opt, yshift_opt, rot_angle_opt, opt_indices, diff_measure_images

    def shift_rot_image(self, image, xshift=0., yshift=0., rot_angle=0., rot_origin=None, invert_transform=False):

        image_out = np.zeros_like(image, 'f4')
        image_in_cl = cl.Image(self.compute_device,
                                   cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                   arr=image.astype('f4'))
        image_out_cl = cl.Image(self.compute_device,
                                cl.Buffer.MemFlags.WRITE_ONLY | cl.Buffer.MemFlags.HOST_READ_ONLY,
                                cl.im_format_float, image.shape)
        rot_angle = np.float32(np.deg2rad(rot_angle))
        shift_vec = np.array((xshift, yshift), 'f4')
        if rot_origin is None:
            rot_origin = (np.array(image.shape, 'f4')[::-1] / 2)
        else:
            rot_origin = np.array(rot_origin, 'f4')

        if invert_transform:
            self.cl_program.shiftrot_image_inv(image.shape[::-1], self.local_size_2d,
                                               image_in_cl, image_out_cl, shift_vec, rot_origin, rot_angle)
        else:
            self.cl_program.shiftrot_image(image.shape[::-1], self.local_size_2d,
                                           image_in_cl, image_out_cl, shift_vec, rot_origin, rot_angle)
        image_out_cl.copy_to(image_out)
        return image_out

    # helpers
    @staticmethod
    def filter_highpass(image, relative_cutoff, smooth_range=None):
        raise NotImplementedError
        highpass = np.ones(500, 'f4')
        cut_index = int(relative_cutoff * 500 + 0.5)
        highpass[:cut_index] = 0.
        if smooth_range is None:
            smooth_range = min(min(cut_index, 500 - cut_index) / 6, 20)
        else:
            smooth_range = smooth_range / 500
        highpass = gaussian_filter1d(highpass, smooth_range, truncate=4)

        u_1d = np.linspace(0, 0.5, 500)
        u_2d = np.sqrt(np.fft.fftfreq(image.shape[0])[:, None] ** 2 + np.fft.rfftfreq(image.shape[1])[None, :] ** 2)
        highpass_2d = get_filter2d_from_radial(u_1d, highpass, u_2d)

        image_ft = np.fft.rfft2(image)
        image_ft *= highpass_2d
        image_filt = np.fft.irfft2(image_ft)
        return image_filt


    @staticmethod
    def slice_to_nonzero(image_scaleshifted):
        y_mean = np.mean(image_scaleshifted, axis=0)
        x_mean = np.mean(image_scaleshifted, axis=1)
        y_indices = np.arange(image_scaleshifted.shape[0])[~np.isclose(x_mean, 0.)]
        x_indices = np.arange(image_scaleshifted.shape[1])[~np.isclose(y_mean, 0.)]
        im_slice = slice(y_indices.min(), y_indices.max()), slice(x_indices.min(), x_indices.max())
        return im_slice

    cl_source = cl.load_source(__file__, 'cl', 'registration.c')

    cl_source_masked = cl.load_source(__file__, 'cl', 'registration_mask.c')
