''' written by Maximilian Ullherr, maximilian.ullherr@physik.uni-wuerzburg.de, Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

contains median filters in OpenCl and CT ring filters based on these

License for this code:
Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

#from file_access import *
import numpy as np, os
import common.opencl as cl


# ======== median filter in OpenCl (ring filters and bad pixel filter) ========
# this source is used in cases that do not parallelize well because too few elements need to be computed,
# or if a median filter needs to be computed for non-neighbouring entries (masked median filter)
quickselect_source = cl.load_source(__file__, 'cl', 'quickselect.c')

# this source is used in cases where enough elements need to be computed,
# so that one thread can compute many elements without loss of parallelization
update_heapsort_source = cl.load_source(__file__, 'cl', 'heapsort.c')


class MedianFilter2D:
    """
    median filter implementation in OpenCl for 2D images

    - optimized only for small sizes (< 50 filter footprint samples)
    - only works for an uneven number of samples (correct filter footprints have 1+4*n samples)

    uses a quickselect algorithm with a median of 3 pivot
    """

    local_size2 = 4, 16  # some filter sizes are much (2x) faster for smaller work group sizes
    local_size1 = 64,

    def __init__(self, compute_device):
        self.compute_device = compute_device

        self.image_cl = None
        self.result_cl = None
        self.offsets_cl = None
        self.num_offsets = None

        self.allocated_shape = (0, 0)
        self.allocated_mask_shape = (0, 0)

        self.result = None
        self.im_format_float = cl.Image.Format(cl.Image.ChannelOrder.INTENSITY, cl.Image.ChannelType.FLOAT)
        self.im_format_uint8 = cl.Image.Format(cl.Image.ChannelOrder.INTENSITY,
                                               cl.Image.ChannelType.UNSIGNED_INT8)

        self.cl_program = cl.Program(self.compute_device, quickselect_source + self.source,
                                         blocking_kernel_calls=True)

        self.threshold = None
        self.global_size2 = None

    # ===== median filter =====
    def allocate_image(self, image: np.ndarray):
        image = image.astype('f4', copy=not image.flags.owndata)
        if not np.allclose(self.allocated_shape, image.shape):
            self.image_cl = cl.Image(self.compute_device,
                                     cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                     arr=image)
            self.result_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.WRITE_ONLY, self.im_format_float,
                                      image.shape)
            self.allocated_shape = image.shape
            self.global_size2 = self.allocated_shape
            self.result = np.zeros(self.allocated_shape, 'f4')
        else:
            self.image_cl.copy_from(image)

    def set_footprint(self, footprint: np.ndarray):
        if footprint is not None:
            self.number_samples = np.int32(footprint.sum())
            assert 3 <= self.number_samples < 256, f'the implementation is limited to 3-256 median samples, has{self.number_samples} (use sparse sampling for higher ranges)'
            offsets = np.zeros(2 * self.number_samples, 'i1')
            k_flat = 0
            center = np.array(footprint.shape) // 2
            for indices, value in np.ndenumerate(footprint):
                if value:
                    offsets[k_flat:k_flat + 2] = (indices - center)[::-1]
                    k_flat += 2

            self.offsets_cl = cl.Buffer(self.compute_device,
                                        cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                        arr=offsets)
        else:
            assert self.offsets_cl is not None, 'must either give a footprint or use set_footprint()'

    def median_filter(self, image: np.ndarray, footprint: np.ndarray = None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)

        self.cl_program.median_filter(self.global_size2, self.local_size2,
                                      self.image_cl, self.result_cl, self.offsets_cl, self.number_samples)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def median_filter_base(self, image_cl, result_cl, wait=False):
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape
        event = self.cl_median_filter(self.global_size2, self.local_size2,
                                      image_cl, result_cl, self.offsets_cl, self.number_samples)
        if wait:
            event.wait()
        return result_cl, image_cl

    def copy_benchmark(self, image: np.ndarray):
        self.allocate_image(image)
        result = self.result
        self.result_cl.copy_to(result)
        return result

    # ===== conditioned median filter =====
    def conditioned_median_filter(self, image: np.ndarray, threshold=None, footprint=None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)
        if threshold is None:
            threshold = self.threshold

        self.cl_program.conditioned_median_filter(self.global_size2, self.local_size2,
                                                  self.image_cl, self.result_cl, self.offsets_cl, self.number_samples,
                                                  np.float32(threshold))

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def conditioned_median_filter_base(self, image_cl, result_cl, threshold=None, wait=True):
        if threshold is None:
            threshold = self.threshold
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape

        event = self.cl_program.conditioned_median_filter(self.global_size2, self.local_size2,
                                                          image_cl, result_cl, self.offsets_cl, self.number_samples,
                                                          np.float32(threshold))
        if wait:
            event.wait()
        return result_cl, image_cl

    # ===== masked/positions median filter =====
    def allocate_mask(self, mask: np.ndarray):
        if mask is not None:
            if not np.allclose(self.allocated_mask_shape, mask.shape):
                self.mask_cl = cl.Image(self.compute_device,
                                        cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                        arr=mask.astype('u1'))
                self.allocated_mask_shape = mask.shape
            else:
                self.mask_cl.copy_from(mask)
        else:
            if self.mask_cl is None: raise ValueError('must give a mask or use allocate_mask() directly')

    def masked_median_filter(self, image: np.ndarray, mask: np.ndarray = None, footprint: np.ndarray = None,
                             allocate_result=False):
        self.allocate_image(image)
        self.allocate_mask(mask)
        self.set_footprint(footprint)

        self.cl_program.masked_median_filter(self.global_size2, self.local_size2,
                                             self.image_cl, self.result_cl, self.offsets_cl, self.number_samples,
                                             self.mask_cl)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def masked_median_filter_base(self, image_cl, result_cl, wait=True):
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape
        event = self.cl_program.masked_median_filter(self.global_size2, self.local_size2,
                                                     image_cl, result_cl, self.offsets_cl, self.number_samples,
                                                     self.mask_cl)
        if wait:
            event.wait()
        return result_cl, image_cl

    def set_positions(self, positions: np.ndarray = None, mask: np.ndarray = None):
        if positions is not None:
            self.length_positions_cl = cl.Buffer(self.compute_device,
                                                 cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                                 arr=np.uint32(positions.shape[0]))
            self.positions_cl = cl.Buffer(self.compute_device,
                                          cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                          arr=positions)
        elif mask is not None:
            self.allocate_mask(mask)

            self.length_positions = np.zeros(1, 'u4')
            self.length_positions_cl = cl.Buffer(self.compute_device,
                                                 cl.Buffer.MemFlags.READ_WRITE | cl.Buffer.MemFlags.COPY_HOST_PTR,
                                                 arr=self.length_positions)
            self.cl_sum_mask(self.global_size2, self.local_size2,
                             self.mask_cl, self.length_positions_cl)
            self.length_positions_cl.copy_from(self.length_positions)
            self.positions_cl = cl.Buffer(self.compute_device, cl.Buffer.MemFlags.READ_ONLY,
                                          # | opencl.Buffer.MemFlags.HOST_NO_ACCESS,
                                          size=4 * self.length_positions[0])
            self.length_positions_cl.copy_from(np.zeros(1, 'u4'))
            self.cl_compute_positions(self.global_size2, self.local_size2,
                                      self.mask_cl, self.positions_cl, self.length_positions_cl)
            self.global_size1 = self.length_positions[::-1]

            # positions = zeros((self.length_positions[0], 2), 'u2')
            # cl.enqueue_copy(self.compute_device, positions, self.positions_cl)
            # print('positions', positions[:20])
        else:
            if self.positions_cl is None:
                raise ValueError('must either give positions or a mask')

    def pos_median_filter(self, image: np.ndarray, positions: np.ndarray = None, mask: np.ndarray = None,
                          footprint: np.ndarray = None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)
        self.set_positions(positions=positions, mask=mask)
        # self.cl_copy_image(self.compute_device, self.global_size2, self.local_size2,
        #                   self.image_cl, self.result_cl)
        # cl.enqueue_copy(self.compute_device, self.result_cl, self.image_cl,
        #                src_origin=(0, 0), dest_origin=(0, 0), region=self.allocated_shape[::-1])
        self.result_cl.copy_from(self.image_cl)

        self.cl_program.pos_median_filter(self.global_size1, self.local_size1,
                                          self.image_cl, self.result_cl,
                                          self.offsets_cl, self.number_samples,
                                          self.positions_cl, self.length_positions_cl)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def pos_median_filter_base(self, image_cl, result_cl, wait=True):
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape
        self.result_cl.copy_from(self.image_cl)

        event = self.cl_program.pos_median_filter(self.global_size1, self.local_size1,
                                                  image_cl, result_cl,
                                                  self.offsets_cl, self.number_samples,
                                                  self.positions_cl, self.length_positions_cl)
        if wait:
            event.wait()
        return result_cl, image_cl

    source = cl.load_source(__file__, 'cl', 'median_filter_2D.c')


class SinoBadPixelFilter(MedianFilter2D):
    local_size2 = 2, 32  # some filter sizes are much (2x) faster for smaller work group sizes

    def __init__(self, compute_device):
        super().__init__(compute_device)

        self.bad_pixel_map_cl = None
        self.allocated_bad_pixel_map_shape = (0, 0)

    def set_bad_pixel_map(self, bad_pixel_map, radius_addition=2):
        # this function allocates the bad pixel map and applies a width transform to the bad_pixel_map
        # if the value of the bad_pixel_map is greater than the width transform, its value is used directly
        if bad_pixel_map is not None:
            if not np.allclose(self.allocated_bad_pixel_map_shape, bad_pixel_map.shape):
                # print('mean of mask', bad_pixel_map.mean(), bad_pixel_map.min(), bad_pixel_map.max())
                bad_pixel_map_cl = cl.Image(self.compute_device,
                                            cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                            self.im_format_uint8, None, arr=bad_pixel_map.astype('u1'))

                self.bad_pixel_map_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE,
                                                 self.im_format_uint8, bad_pixel_map.shape)
                self.cl_program.mask_width_transform((bad_pixel_map.shape[0],), (64,),
                                                     bad_pixel_map_cl, self.bad_pixel_map_cl, np.int32(radius_addition))
                # cl.enqueue_copy(self.compute_device, bad_pixel_map, self.bad_pixel_map_cl, origin=(0, 0), region=tuple(bad_pixel_map.shape[::-1]))
                # print('mean of radial transformed mask', bad_pixel_map.mean(), bad_pixel_map.min(), bad_pixel_map.max())

                self.allocated_bad_pixel_map_shape = bad_pixel_map.shape
            else:
                self.bad_pixel_map_cl.copy_from(bad_pixel_map)
        else:
            if self.bad_pixel_map_cl is None: raise ValueError(
                'must give a bad_pixel_map or use allocate_bad_pixel_map() directly')

    def filter_bad_pixels_sino(self, image: np.ndarray, mask: np.ndarray,
                               mask_y_pos, allocate_result=False):
        self.allocate_image(image)
        self.set_bad_pixel_map(mask)

        self.cl_program.filter_bad_pixels_sino(self.global_size2, self.local_size2,
                                               self.image_cl, self.result_cl, self.bad_pixel_map_cl, np.int32(mask_y_pos))

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.write_to(result)
        return result

    def filter_bad_pixels_sino_base(self, image_cl, result_cl, mask_y_pos, wait=True):
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape
        assert self.bad_pixel_map_cl is not None, 'must use set_bad_pixel_map() before'
        event = self.cl_program.filter_bad_pixels_sino(self.global_size2, self.local_size2,
                                                       image_cl, result_cl, self.bad_pixel_map_cl, np.int32(mask_y_pos))
        if wait:
            event.wait()
        return result_cl, image_cl

    def set_pos1d(self, mask: np.ndarray, y_pos):
        self.allocate_mask(mask)

        self.length_positions = np.zeros(1, 'u4')
        self.length_positions_cl = cl.Buffer(self.compute_device,
                                             cl.Buffer.MemFlags.READ_WRITE | cl.Buffer.MemFlags.COPY_HOST_PTR,
                                             arr=self.length_positions)
        self.global_size1 = mask.shape[1:][::-1]
        self.cl_program.sum_mask1d(self.global_size1, self.local_size1,
                                   self.mask_cl, self.length_positions_cl, np.int32(y_pos))
        self.length_positions_cl.copy_to(self.length_positions)
        self.positions_cl = cl.Buffer(self.compute_device, cl.Buffer.MemFlags.READ_ONLY,
                                      # | opencl.Buffer.MemFlags.HOST_NO_ACCESS,
                                      size=2 * self.length_positions[0])
        self.length_positions_cl.copy_from(np.zeros(1, 'u4'))
        self.cl_program.compute_pos1d(self.global_size1, self.local_size1,
                                      self.mask_cl, np.int32(y_pos), self.positions_cl, self.length_positions_cl)
        self.global_size1 = self.length_positions[::-1]

        # positions = zeros(self.length_positions[0], 'u2')
        # cl.enqueue_copy(self.compute_device, positions, self.positions_cl)
        # print('positions', sorted(positions))

    def pos1d_median_filter(self, image: np.ndarray, mask: np.ndarray, y_pos,
                            allocate_result=False):
        print('WARNING: this code is not intended to be executed')
        self.allocate_image(image)
        self.set_pos1d(mask=mask, y_pos=y_pos)
        self.result_cl.copy_from(self.image_cl)

        local_size = (32, 4)
        self.cl_program.pos1d_median_filter((self.allocated_shape[0], self.global_size1[0]), local_size,
                                            self.image_cl, self.result_cl, self.mask_cl,
                                            self.positions_cl, self.length_positions_cl)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def pos1d_median_filter_base(self, image_cl, result_cl, wait=True):
        print('WARNING: this code is not intended to be executed')
        if self.global_size2 is None:
            self.global_size2 = image_cl.shape
        result_cl.copy_from(image_cl)

        local_size = (8, 16)
        event = self.cl_program.pos1d_median_filter((self.allocated_shape[0], self.global_size1[0]), local_size,
                                                    image_cl, result_cl, self.mask_cl,
                                                    self.positions_cl, self.length_positions_cl)
        if wait:
            event.wait()
        return result_cl, image_cl

    source = MedianFilter2D.source + cl.load_source(__file__, 'cl', 'bad_pixel_filter.c')


class MedianFilter3D():
    """
    median filter implementation in OpenCl for 3D (volume) images

    - optimized only for small sizes (< 50 filter footprint samples)
    - only works for an uneven number of samples (correct filter footprints have 1+6*n samples)

    uses a quickselect algorithm with the simplest possible pivot choice
    """

    local_size3 = 4, 4, 8  # almost no performance differences for sizes 16-128
    local_size1 = 64,

    def __init__(self, compute_device):
        self.compute_device = compute_device

        self.image_cl = None
        self.result_cl = None
        self.offsets_cl = None
        self.num_offsets = None

        self.allocated_shape = (0, 0, 0)
        self.allocated_mask_shape = (0, 0, 0)

        self.result = None
        self.im_format_float = cl.Image.Format(cl.Image.ChannelOrder.INTENSITY, cl.Image.ChannelType.FLOAT)
        self.im_format_uint8 = cl.Image.Format(cl.Image.ChannelOrder.INTENSITY,
                                               cl.Image.ChannelType.UNSIGNED_INT8)

        self.cl_program = cl.Program(self.compute_device, self.source)

    # ===== median filter =====
    def allocate_image(self, image: np.ndarray):
        image = image.astype('f4', copy=not image.flags.owndata)
        if not np.allclose(self.allocated_shape, image.shape):
            self.image_cl = cl.Image(self.compute_device,
                                     cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                     arr=image)
            self.result_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.WRITE_ONLY, self.im_format_float,
                                      image.shape)
            self.allocated_shape = image.shape
            self.global_size3 = self.allocated_shape
            self.result = np.zeros(self.allocated_shape, 'f4')
        else:
            self.image_cl.copy_from(image)

    def set_footprint(self, footprint: np.ndarray):
        if footprint is not None:
            self.number_samples = np.int32(footprint.sum())
            assert 3 < self.number_samples < 256, 'the implementation is limited to 3-256 median samples (use sparse sampling for higher ranges)'
            offsets = np.zeros(4 * self.number_samples, 'i1')
            k_flat = 0
            center = np.array(footprint.shape) // 2
            for indices, value in np.ndenumerate(footprint):
                if value:
                    offsets[k_flat:k_flat + 3] = (indices - center)[::-1]
                    k_flat += 4

            self.offsets_cl = cl.Buffer(self.compute_device,
                                        cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                        arr=offsets)
        else:
            assert self.offsets_cl is not None, 'must either give a footprint or use set_footprint()'

    def median_filter(self, image: np.ndarray, footprint: np.ndarray = None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)

        self.cl_program.median_filter(self.global_size3, self.local_size3,
                                      self.image_cl, self.result_cl, self.offsets_cl, self.number_samples)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def median_filter_base(self, image_cl, result_cl, wait=True):
        event = self.cl_program.median_filter(self.global_size3, self.local_size3,
                                              image_cl, result_cl, self.offsets_cl, self.number_samples)
        if wait:
            event.wait()

    def copy_benchmark(self, image: np.ndarray):
        self.allocate_image(image)
        self.result_cl.copy_to(self.result)
        return self.result

    # ===== conditioned median filter =====
    def conditioned_median_filter(self, image: np.ndarray, threshold, footprint=None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)

        self.cl_program.conditioned_median_filter(self.global_size3, self.local_size3,
                                                  self.image_cl, self.result_cl, self.offsets_cl, self.number_samples,
                                                  np.float32(threshold))

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(self.result)
        return result

    def conditioned_median_filter_base(self, threshold, image_cl, result_cl, wait=True):
        event = self.cl_program.conditioned_median_filter(self.global_size3, self.local_size3,
                                                          image_cl, result_cl, self.offsets_cl, self.number_samples,
                                                          np.float32(threshold))
        if wait:
            event.wait()

    # ===== masked/positions median filter =====
    def allocate_mask(self, mask: np.ndarray):
        if mask is not None:
            if not np.allclose(self.allocated_mask_shape, mask.shape):
                mask = mask.astype('u1')
                self.mask_cl = cl.Image(self.compute_device,
                                        cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_WRITE_ONLY,
                                        arr=mask)
                self.allocated_mask_shape = mask.shape
                self.global_size3 = self.allocated_mask_shape
            else:
                self.mask_cl.copy_from(mask)
        else:
            if self.mask_cl is None: raise ValueError('must give a mask or use allocate_mask() directly')

    def masked_median_filter(self, image: np.ndarray, mask: np.ndarray = None, footprint: np.ndarray = None,
                             allocate_result=False):
        self.allocate_image(image)
        self.allocate_mask(mask)
        self.set_footprint(footprint)

        self.cl_program.masked_median_filter(self.global_size3, self.local_size3,
                                             self.image_cl, self.result_cl, self.offsets_cl, self.number_samples,
                                             self.mask_cl)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def masked_median_filter_base(self, image_cl, result_cl, wait=True):
        event = self.cl_program.masked_median_filter(self.global_size3, self.local_size3,
                                                     image_cl, result_cl, self.offsets_cl, self.number_samples,
                                                     self.mask_cl)
        if wait:
            event.wait()

    def set_positions(self, positions: np.ndarray = None, mask: np.ndarray = None):
        if positions is not None:
            self.length_positions_cl = cl.Buffer(self.compute_device,
                                                 cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                                 arr=np.uint32(positions.shape[0]))
            self.positions_cl = cl.Buffer(self.compute_device,
                                          cl.Buffer.MemFlags.READ_ONLY | cl.Buffer.MemFlags.COPY_HOST_PTR | cl.Buffer.MemFlags.HOST_NO_ACCESS,
                                          arr=positions)
        elif mask is not None:
            self.allocate_mask(mask)

            self.length_positions = np.zeros(1, 'u4')
            self.length_positions_cl = cl.Buffer(self.compute_device,
                                                 cl.Buffer.MemFlags.READ_WRITE | cl.Buffer.MemFlags.COPY_HOST_PTR,
                                                 arr=self.length_positions)
            self.cl_program.sum_mask(self.global_size3, self.local_size3,
                                     self.mask_cl, self.length_positions_cl)
            self.length_positions_cl.copy_to(self.length_positions)
            self.positions_cl = cl.Buffer(self.compute_device, cl.Buffer.MemFlags.READ_ONLY,
                                          # | opencl.Buffer.MemFlags.HOST_NO_ACCESS,
                                          size=8 * self.length_positions[0])
            self.length_positions_cl.copy_from(np.zeros(1, 'u4'))
            global_size = mask.shape[::-1]
            self.cl_program.compute_positions(self.global_size3, self.local_size3,
                                              self.mask_cl, self.positions_cl, self.length_positions_cl)
            self.global_size1 = self.length_positions[::-1]

            # positions = zeros((self.length_positions[0], 2), 'u2')
            # self.positions_cl.read_to(positions)
            # print('positions', positions[:20])
        else:
            if self.positions_cl is None:
                raise ValueError('must either give positions or a mask')

    def pos_median_filter(self, image: np.ndarray, positions: np.ndarray = None, mask: np.ndarray = None,
                          footprint: np.ndarray = None, allocate_result=False):
        self.allocate_image(image)
        self.set_footprint(footprint)
        self.set_positions(positions=positions, mask=mask)
        self.result_cl.copy_from(self.image_cl)

        self.cl_program.pos_median_filter(self.global_size1, self.local_size1,
                                          self.image_cl, self.result_cl,
                                          self.offsets_cl, self.number_samples,
                                          self.positions_cl, self.length_positions_cl)

        if allocate_result:
            result = np.zeros(self.allocated_shape, 'f4')
        else:
            result = self.result
        self.result_cl.copy_to(result)
        return result

    def pos_median_filter_base(self, image_cl, result_cl, wait=False):
        self.result_cl.copy_from(image_cl)
        event = self.cl_program.pos_median_filter(self.global_size1, self.local_size1,
                                                  image_cl, result_cl,
                                                  self.offsets_cl, self.number_samples,
                                                  self.positions_cl, self.length_positions_cl)
        if wait:
            event.wait()

    source = cl.load_source(__file__, 'cl', 'median_filter_3D.c')
