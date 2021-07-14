''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
from image.median_filters import *

class SinoRingFilterer(MedianFilter2D):
    ring_local_size = 128,

    def __init__(self, compute_device):
        super().__init__(compute_device)

        self.line_images_cl = None
        self.allocated_width = 0

    def allocate_line_images(self, width):
        if self.allocated_width != width:
            self.line_images_cl = [
                cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE, self.im_format_float, (width,)) for
                k in (0, 1)]
            self.allocated_width = width
            self.ring_global_size = (width,)

    def filter_rings_base(self, image_cl, result_cl, radius, wait=True):
        self.allocate_line_images(image_cl.shape[1])
        self.cl_program.ring_filter_prepare(self.ring_global_size, self.ring_local_size,
                                            image_cl, self.line_images_cl[0]).wait()
        self.cl_program.ring_filter_median(self.ring_global_size, self.ring_local_size,
                                           self.line_images_cl[0], self.line_images_cl[1], np.int32(radius))

        event = self.cl_program.ring_filter_apply(self.ring_global_size, self.ring_local_size,
                                                  image_cl, result_cl, self.line_images_cl[1])
        if wait:
            event.wait()
        return result_cl, image_cl

    source_ringf = """
    float median_of_3(float val1, float val2, float val3){
        if (val1 > val2){
            if (val1 < val3){return(val1);}
            else {
                if (val2 > val3){return(val2);}
                else {return(val3);}
                }
            }
        else {
            if (val2 < val3){return(val2);}
            else {
                if (val1 > val3){return(val1);}
                else {return(val3);}
                }
            }
        }

    kernel void ring_filter_prepare(read_only image2d_t image, write_only image1d_t y_mean){ 
        const int pos = get_global_id(0);
        const int2 shape = get_image_dim(image);
        if (pos < shape.x){
            float sum_val = 0.0f;
            for (ushort k=0; k < shape.y; k++){
                sum_val += read_imagef(image, sampler, (int2)(pos, k)).x;
                }
            write_imagef(y_mean, pos, (float4)(sum_val/shape.y, 0, 0, 0));
            }
        }

    kernel void ring_filter_median(read_only image1d_t y_mean_in, write_only image1d_t y_mean_out,
                                   int radius){ 
        float values[255];
        const int pos = get_global_id(0);
        const int shape = get_image_width(y_mean_out);
        if (pos < shape){
            float value = read_imagef(y_mean_in, sampler, pos).x;
            int length = 1+2*radius;
            for (short k=0; k < length; k++){
                values[k] = read_imagef(y_mean_in, sampler, pos+k-radius).x;
                }
            value -= quickselect_median(values, 1+2*radius);
            write_imagef(y_mean_out, pos, (float4)(value, 0, 0, 0));
            }
        }

    kernel void ring_filter_apply(read_only image2d_t image, write_only image2d_t result,
                            read_only image1d_t y_mean){    
        const int2 pos = (int2)(get_global_id(0), 0);
        const int2 shape = (int2)(get_image_width(image), get_image_height(image));
        if (pos.x < shape.x){
            float correction = read_imagef(y_mean, sampler, pos.x).x;
            for (ushort k=0; k < shape.y; k++){
                pos.y = k;
                write_imagef(result, pos, (float4)((read_imagef(image, sampler, pos).x - correction), 0, 0, 0));
                }
            }
        } 

    """

    source = MedianFilter2D.source + source_ringf
    

class PolarTransform:
    radial_local_size = 128,
    cartsian_local_size = 16, 8
    polar_local_size = 16, 8

    def __init__(self, compute_device):
        self.compute_device = compute_device

        self.image_cl = None
        self.result_cl = None
        self.center = None

        self.cl_program_polar = cl.Program(self.compute_device, self.polar_transform_source)

        self.allocated_cartesian_shape = (0, 0)
        self.allocated_polar_shape = (0, 0)
        self.sampling_factors = np.array((3.5, 1.5), "f4")
        # this is a very high oversampling, it results in almost all pixels being correct within float precision

    def allocate_images(self, image, allocate_polar=True):
        if not np.allclose(image.shape, self.allocated_cartesian_shape):
            self.image_cl = cl.Image(self.compute_device,
                                     cl.Buffer.MemFlags.READ_WRITE | cl.Buffer.MemFlags.COPY_HOST_PTR,
                                     arr=image)
            self.allocated_cartesian_shape = image.shape
            self.cartesian_work_size = image.shape
            self.image = np.zeros(image.shape, 'f4')
        else:
            self.image_cl.copy_from(image)

        if allocate_polar:
            polar_shape = self.polar_shape(image.shape)
            if not np.allclose(polar_shape, self.allocated_polar_shape):
                self.image_polar_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE,
                                               cl.im_format_float,
                                               polar_shape)
                self.allocated_polar_shape = polar_shape
                self.radial_work_size = (self.allocated_polar_shape[1],)
                self.polar_work_size = self.allocated_polar_shape

    def polar_shape(self, image_shape):
        max_diameter = np.sqrt(sum(np.asarray(image_shape) ** 2))
        polar_shape = np.asarray((max_diameter * np.pi * self.sampling_factors[1],
                               max_diameter / 2 * self.sampling_factors[0] + 4), "i4")
        polar_shape += polar_shape % 2
        return polar_shape

    def polar_transform_forward(self, image, center=None, sampling_factors=None):
        if sampling_factors is not None:
            self.sampling_factors = np.asarray(sampling_factors, "f4")  # sqrt for diagonal sampling (both angular and radial)

        self.allocate_images(image)
        if center is not None:
            self.center = center
        if self.center is None:
            self.center = (np.array(image.shape[::-1], dtype='f4') - 1) / 2

        self.cl_program_polar.polar_forward_transform(self.polar_work_size, self.polar_local_size,
                                                      self.image_cl, self.image_polar_cl,
                                                      self.center, self.sampling_factors)

    def image_polar(self):
        image_polar = np.zeros(self.allocated_polar_shape, 'f4')
        self.image_polar_cl.copy_to(image_polar)
        return image_polar

    def polar_transform_backward(self, polar_image=None, return_arr=False):
        if polar_image is not None:
            assert 'f4' in polar_image.dtype.str, 'image must be float32'
            self.image_polar_cl.copy_from(polar_image)

        self.cl_program_polar.polar_backward_transform(self.cartesian_work_size, self.cartsian_local_size,
                                                       self.image_polar_cl, self.image_cl, self.center,
                                                       self.sampling_factors)

        self.image_cl.copy_to(self.image)
        if return_arr:
            return self.image.copy()

    polar_transform_source = '''
    constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
    constant sampler_t wrap_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;

    kernel void polar_forward_transform(read_only image2d_t image,
                                        write_only image2d_t image_polar,
                                        float2 center, float2 sampling_factors) {
        const int2 pos_polar = (int2)(get_global_id(0), get_global_id(1)); // x=radial, y=angular
        const int2 shape_polar = get_image_dim(image_polar);

        if (all(pos_polar < shape_polar)){
            const float2 coord_polar = (float2)((float)(pos_polar.x)/sampling_factors.x, 
                                                (float)(pos_polar.y)/(float)(shape_polar.y)*2.f*M_PI_F);
            const int2 coord_cart = convert_int2(center + coord_polar.x*(float2)(cos(coord_polar.y), sin(coord_polar.y)) + 0.5f);
            write_imagef(image_polar, pos_polar, read_imagef(image, nearest_sampler, coord_cart));
            }
        }

    kernel void polar_backward_transform(read_only image2d_t image_polar,
                                         write_only image2d_t image,
                                         float2 center, float2 sampling_factors) {
        const int2 pos_cart = (int2)(get_global_id(0), get_global_id(1));
        const int2 shape_cart = get_image_dim(image);

        if (all(pos_cart < shape_cart)){
            const int2 shape_polar = get_image_dim(image_polar);
            const float2 coord_cart = convert_float2(pos_cart)-center;
            const float2 coord_polar_norm = (float2)((length(coord_cart)*sampling_factors.x+0.5f) / (float)shape_polar.x,
                                                     atan2pi(coord_cart.y, coord_cart.x)*0.5f + 0.5f/(float)shape_polar.y);
            if (coord_polar_norm.x < 1.f){ 
                write_imagef(image, pos_cart, read_imagef(image_polar, wrap_sampler, coord_polar_norm));
                }
            }
        }
    '''


class SliceFullRingsFilterer(PolarTransform):
    def __init__(self, compute_device: cl.ComputeDevice):
        super().__init__(compute_device)
        self.allocated_full_correction_length = 0

        self.cl_program_full_rings = cl.Program(self.compute_device, quickselect_source, self.source_full_rings,
                                                blocking_kernel_calls=True)

    # full rings
    def allocate_full_rings(self, image_shape):
        polar_radial_length = self.polar_shape(image_shape)[1]
        if self.allocated_full_correction_length != polar_radial_length:
            self.angle_median_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE, cl.im_format_float,
                                            (polar_radial_length,))
            self.ring_correction_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE,
                                               cl.im_format_float, (polar_radial_length,))
            self.allocated_full_correction_length = polar_radial_length
            self.radial_work_size = (polar_radial_length,)

    def generate_weigthing_median_offsets(self, width: float, exp_factor=2.):
        # higher exp_factor leads to longer range + sparser sampling
        stride = width ** 0.2
        force = exp_factor / width
        x = np.arange(width)
        offsets = stride * (np.exp(x * force) - 1) / force
        return offsets.astype('f4'), np.uint16(len(offsets))

    def filter_rings(self, image, full_rings_max_width,
                     full_sampling_factor_radial=1.4142, full_sampling_factor_angular=0.1,
                     center_shift_x=0., center_shift_y=0.,
                     use_gray_level=False, gray_level_threshold=0.5, gray_level=1.0, **kwargs):
        self.center = (np.array(image.shape[::-1], dtype='f4') - 1) / 2 + np.array((center_shift_x, center_shift_y), 'f4')
        self.sampling_factors = np.array((full_sampling_factor_radial, full_sampling_factor_angular), 'f4')
        self.sampling_factors *= 1.4142  # diagonal sampling

        image = image.astype('f4')
        if use_gray_level:
            loc = image > gray_level_threshold
            image[loc] -= gray_level
        self.allocate_full_rings(image.shape)

        self.allocate_images(image, allocate_polar=False)
        #t0 = time.time()
        self.cl_program_full_rings.angular_median(self.radial_work_size, self.radial_local_size,
                                                  self.image_cl, self.angle_median_cl,
                                                  self.center, self.sampling_factors)
        #print_runtime('angular median runtime', t0); t0 = time.time()
        if full_rings_max_width > 0:
            median_size = np.int32(self.sampling_factors[0] * full_rings_max_width * 2 + 1)
            #stride = int32(median_size // 511 + 1)
            stride = np.int32(np.fmax(np.int32((median_size/75)**0.5)+1, median_size//511+1))  # less samples above 75
            median_size = np.int32(median_size / stride)
            self.cl_program_full_rings.detrend_correction(self.radial_work_size, self.radial_local_size,
                                                          self.angle_median_cl, self.ring_correction_cl,
                                                          median_size, stride)
            self.cl_program_full_rings.generate_correction_image(self.cartesian_work_size, self.cartsian_local_size,
                                                                 self.ring_correction_cl, self.image_cl,
                                                                 self.center, self.sampling_factors)
        else:
            self.cl_program_full_rings.generate_correction_image(self.cartesian_work_size, self.cartsian_local_size,
                                                                 self.angle_median_cl, self.image_cl,
                                                                 self.center, self.sampling_factors)

        #print_runtime('limit ring width runtime', t0); t0 = time.time()
        self.image_cl.copy_to(self.image)
        image -= self.image
        #print_runtime('result copy', t0); t0 = time.time()

        if use_gray_level:
            image[loc] += gray_level

        return image

    source_full_rings = """
    constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_LINEAR;
    constant sampler_t wrap_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR; 
    constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // code for full rings filter
    // fast implementation without polar transforms (2d cartesian => 1d radial => 2d cartesian)  
    kernel void angular_median(read_only image2d_t image, write_only image1d_t correction, 
                               float2 center, float2 sampling_factors){
        const int pos_x = get_global_id(0);
        const int radial_length = get_image_width(correction);
        if (pos_x < radial_length){
            float2 coord_cart; 
            float values[511];
            float angle, radius = pos_x/sampling_factors.x;
            int med_length = min((int)(2.f*M_PI_F*radius*sampling_factors.y) + 5, 511);
            med_length -= (med_length % 2) + 1;
            for (int k=0; k < med_length; k++){
                angle = 2.f*M_PI_F*(float)k / (float)med_length;
                coord_cart = center + radius*(float2)(cos(angle), sin(angle)) + 0.5f;
                values[k] = read_imagef(image, sampler, coord_cart).x;
                }
            write_imagef(correction, pos_x, (float4)(quickselect_median(values, med_length)));
            }
        }

    kernel void detrend_correction(read_only image1d_t correction_in, write_only image1d_t correction_out, 
                                   int med_length, int stride){
        const int pos_r = get_global_id(0);
        if (pos_r < get_image_width(correction_in)){
            //write_imagef(correction_out, pos_r, (float4)(0.f)); 
            if (med_length >= 3){
                med_length = min(511, med_length);
                float values[511];
                int read_pos, offset = med_length/2;
                for (int k=0; k <= med_length; k++){
                    read_pos = abs(pos_r + (k - offset)*stride);  // abs() mirrors at zero
                    values[k] = read_imagef(correction_in, sampler, read_pos).x;
                    }
                float4 value = read_imagef(correction_in, sampler, pos_r);
                value.x -= quickselect_median(values, med_length);
                write_imagef(correction_out, pos_r, value); 
                }
            else{
                write_imagef(correction_out, pos_r, read_imagef(correction_in, sampler, pos_r)); 
                }
            }
        } 

    kernel void generate_correction_image(read_only image1d_t correction, write_only image2d_t image,
                                         float2 center, float2 sampling_factors){
        const int2 pos = (int2)(get_global_id(0), get_global_id(1));
        const int2 shape = get_image_dim(image);
        if (all(pos < shape)){
            float radius = length(convert_float2(pos)-center)*sampling_factors.x+0.5f;
            write_imagef(image, pos, read_imagef(correction, linear_sampler, radius));  
            }
        }
    """


class SlicePartialRingsFilterer(PolarTransform):
    def __init__(self, compute_device: cl.ComputeDevice, max_median_samples=511):
        super().__init__(compute_device)
        self.allocated_partial_shape = (0, 0)
        self.max_median_samples = max_median_samples

        self.cl_program_partial_rings = cl.Program(self.compute_device, update_heapsort_source, self.source_partial_rings,
                                                   fill_placeholders=(('max_median_samples', max_median_samples),),
                                                   blocking_kernel_calls=True)

    # partial rings
    def allocate_partial_rings(self):
        if not np.allclose(self.allocated_partial_shape, self.allocated_polar_shape):
            self.image_polar_2_cl = cl.Image(self.compute_device, cl.Buffer.MemFlags.READ_WRITE,
                                             cl.im_format_float, self.allocated_polar_shape)
            self.allocated_partial_shape = self.allocated_polar_shape

    def compute_work_sizes_angular_median(self, median_filter_size, bunch_axis):
        median_filter_size = min(median_filter_size, 511)
        bunch_size = self.allocated_polar_shape[bunch_axis] // (
                    self.allocated_polar_shape[bunch_axis] // (median_filter_size * 20) + 1) + 1
        # bunch size is at most 20*filt_size and image.shape[0] is slightly smaller than a multiple of filt_size
        global_size = np.array(self.allocated_polar_shape, 'i4')
        global_size[bunch_axis] = (global_size[bunch_axis] // bunch_size + 1)
        local_size_skew_fact = np.fmax(np.int32(2 ** (3.0 - np.log2(global_size[bunch_axis]))), 1)
        if bunch_axis == 0:
            local_size = (8 // local_size_skew_fact, 8 * local_size_skew_fact)
        else:
            local_size = (8 * local_size_skew_fact, 8 // local_size_skew_fact)
        #print('bunch work sizes', global_size, local_size, int32(bunch_size))
        return global_size, local_size, np.int32(bunch_size)

    def swap_polar_image_references(self):
        polar = self.image_polar_cl
        self.image_polar_cl = self.image_polar_2_cl
        self.image_polar_2_cl = polar

    def adjust_angular_sampling(self, image_shape, partial_angular_fraction):
        if partial_angular_fraction > 0.:
            polar_shape = self.polar_shape(image_shape)
            #print('adjust_angular_sampling before', self.sampling_factors[1])
            self.sampling_factors[1] *= np.fmin(self.max_median_samples/partial_angular_fraction/polar_shape[1], 1.)
            #print('adjust_angular_sampling after', self.sampling_factors[1])

    def filter_rings(self, image, partial_rings_max_width,
                     partial_absolute_size, partial_angular_fraction=0.0,
                     partial_sampling_factor_radial=1.4142, partial_sampling_factor_angular=0.1,
                     center_shift_x=0., center_shift_y=0.,
                     use_gray_level=False, gray_level_threshold=0.5, gray_level=1.0, **kwargs):

        self.center = (np.array(image.shape[::-1], dtype='f4') - 1) / 2 + np.array((center_shift_x, center_shift_y), 'f4')
        self.sampling_factors = np.array((partial_sampling_factor_radial, partial_sampling_factor_angular), 'f4')
        self.sampling_factors *= 1.4142  # diagonal sampling
        self.adjust_angular_sampling(image.shape, partial_angular_fraction)

        image = image.astype('f4')
        if use_gray_level:
            loc = image > gray_level_threshold
            image[loc] -= gray_level

        self.polar_transform_forward(image)

        partial_absolute_size = np.fmax(11, partial_absolute_size)  # kernel crashes for very small values
        self.allocate_partial_rings()
        median_length = np.int32(1 + 2 * partial_absolute_size)
        polar_work_size, polar_local_size, bunch_size = self.compute_work_sizes_angular_median(median_length, 0)
        self.cl_program_partial_rings.angular_median_filter(polar_work_size, polar_local_size,
                                                        self.image_polar_cl, self.image_polar_2_cl,
                                                        median_length, np.float32(partial_angular_fraction),
                                                        self.sampling_factors, bunch_size)

        self.cl_program_partial_rings.fill_angular_strides(self.polar_work_size, self.polar_local_size,
                                                           self.image_polar_2_cl, self.image_polar_cl)

        if partial_rings_max_width > 0:
            detrend_length_sampled = np.int32(1 + 2 * int(partial_rings_max_width * self.sampling_factors[0]))
            polar_work_size, polar_local_size, bunch_size = self.compute_work_sizes_angular_median(
                                                                                            detrend_length_sampled, 1)
            self.cl_program_partial_rings.detrend_polar_correction(polar_work_size, polar_local_size,
                                                                   self.image_polar_cl, self.image_polar_2_cl,
                                                                   detrend_length_sampled, bunch_size)
            self.swap_polar_image_references()

        self.polar_transform_backward()
        image -= self.image

        if use_gray_level:
            image[loc] += gray_level
        return image

    source_partial_rings = """
    constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
    constant sampler_t reflect_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_MIRRORED_REPEAT | CLK_FILTER_LINEAR;
    constant sampler_t wrap_sampler = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR; 
    constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

    // code for partial rings filter
    int angular_stride_size(int radius, int radial_length){
        // the median filter is only computed on the strided array, 
        // the intermediate values are then filled in fill_angular_strides()
        return (clamp((int)((float)radial_length/(float)radius*0.75f), 1, radial_length/15));
        //return (1);
        }

    kernel void angular_median_filter(read_only image2d_t polar_image, write_only image2d_t correction,
                                      int min_size, float min_angular_fraction, float2 sampling_factors, 
                                      int bunch_size){
        const int pos_x = get_global_id(0);
        const int2 shape = get_image_dim(correction);
        const float2 shape_inv = 1.f/convert_float2(shape);
        const float radius = (float)pos_x/sampling_factors.x;
        float2 read_pos_norm = (float2)((pos_x+0.5f)*shape_inv.x, 0.f);
        const float pixel_angular_size = (float)shape.x/((float)pos_x + 1.f);
        int med_area = fmax(pixel_angular_size*min_size*sampling_factors.y, 
                            min_angular_fraction*2.f*M_PI_F*radius*sampling_factors.y);
        const int angular_stride = angular_stride_size(pos_x, shape.x);
        int pos_y_start_0 = get_global_id(1)*bunch_size; 
        int pos_y_start = ((pos_y_start_0-1)/angular_stride + 1)*angular_stride*(pos_y_start_0>0); 
        med_area = med_area/angular_stride + 1;

        if (pos_x < shape.x && pos_y_start_0 < shape.y && pos_y_start < (pos_y_start_0+bunch_size)){
            int median_samples = min(med_area, max_median_samples);
            median_samples -= 1 - (median_samples % 2);
            int med_offset = median_samples/2;
            float values[max_median_samples];

            for (int k=0; k < median_samples; k++){
                read_pos_norm.y = ((float)(pos_y_start+(k-med_offset)*angular_stride) + 0.5f)*shape_inv.y;
                values[k] = read_imagef(polar_image, wrap_sampler, read_pos_norm).x;
                }
            heapsort(values, median_samples);    
            write_imagef(correction, (int2)(pos_x, pos_y_start), (float4)(values[med_offset]));

            float val_insert, val_remove;
            int stop = min(pos_y_start+bunch_size, shape.y);
            for (int pos_y=pos_y_start+angular_stride; pos_y < stop; ){
                read_pos_norm.y = ((float)(pos_y+med_offset*angular_stride) + 0.5f)*shape_inv.y;
                val_insert = read_imagef(polar_image, wrap_sampler, read_pos_norm).x;

                read_pos_norm.y = ((float)(pos_y-med_offset*angular_stride-angular_stride) + 0.5f)*shape_inv.y;
                val_remove = read_imagef(polar_image, wrap_sampler, read_pos_norm).x;

                update_sorted(values, median_samples, val_insert, val_remove);
                write_imagef(correction, (int2)(pos_x, pos_y), (float4)(values[med_offset]));
                pos_y+=angular_stride;
                }
            }
        }

    kernel void fill_angular_strides(read_only image2d_t correction_in, write_only image2d_t correction_out){
        const int2 pos = (int2)(get_global_id(0), get_global_id(1));
        const int2 shape = get_image_dim(correction_out);
        if (all(pos<shape)){
            const int angular_stride = angular_stride_size(pos.x, shape.x);
            // no case for before_pos = pos.x because it would not improve performance and is correct as below        
            const int before_pos = pos.y - pos.y%angular_stride;
            int after_pos = before_pos + angular_stride;
            after_pos = min(after_pos, shape.y);  // map small pos to zero (small pos are empty)
            const float value_before = read_imagef(correction_in, nearest_sampler, (int2)(pos.x, before_pos)).x;
            const float value_after  = read_imagef(correction_in, nearest_sampler, (int2)(pos.x, after_pos%shape.y)).x;
            const float value = ( value_before*(float)(after_pos-pos.y) 
                                + value_after*(float)(pos.y-before_pos) 
                                ) / (float)(after_pos-before_pos);
            write_imagef(correction_out, pos, (float4)(value));

            }
        }

    float2 fold_read_pos(float2 read_pos, int2 shape){
        // coordinate folding means that the median filter will pass over the center to the pixels opposite (180 degrees rotated)
        if (read_pos.x < 0.f){
            float2 folded_read_pos;
            folded_read_pos.y = read_pos.y - shape.y/2;
            folded_read_pos.x = -read_pos.x;
            return(folded_read_pos);
            }
        else{return(read_pos);}
        }

    kernel void detrend_polar_correction(read_only image2d_t correction_in, write_only image2d_t correction_out,
                                         int detrend_length, int bunch_size){
        const int pos_x_start = get_global_id(0)*bunch_size;
        const int pos_y = get_global_id(1);
        const int2 shape = get_image_dim(correction_out);

        if (pos_x_start < shape.x && pos_y < shape.y){
            int median_samples = min(detrend_length, max_median_samples);
            median_samples -= 1 - (median_samples % 2);
            float stride = (float)detrend_length/(float)median_samples;
            int median_offset = median_samples/2;
            float2 read_pos = (float2)(0, pos_y)+0.5f, read_pos_folded;
            float values[max_median_samples];
            for (int k=0; k < median_samples; k++){
                read_pos.x = (float)(pos_x_start + k - median_offset)*stride + 0.5f;
                read_pos_folded = fold_read_pos(read_pos, shape);
                values[k] = read_imagef(correction_in, linear_sampler, read_pos_folded).x;
                }
            heapsort(values, median_samples);    
            float4 value = read_imagef(correction_in, nearest_sampler, (int2)(pos_x_start, pos_y));
            value.x -= values[median_offset];
            write_imagef(correction_out, (int2)(pos_x_start, pos_y), value);

            float val_insert, val_remove;
            int stop = min(pos_x_start+bunch_size, shape.x);
            for (int pos_x=pos_x_start+1; pos_x < stop; pos_x++){
                read_pos.x = (float)(pos_x+median_offset)*stride + 0.5f;
                read_pos_folded = fold_read_pos(read_pos, shape);
                val_insert = read_imagef(correction_in, linear_sampler, read_pos_folded).x;

                read_pos.x = (float)(pos_x-median_offset-1)*stride + 0.5f;
                read_pos_folded = fold_read_pos(read_pos, shape);
                val_remove = read_imagef(correction_in, linear_sampler, read_pos_folded).x;

                update_sorted(values, median_samples, val_insert, val_remove);
                read_pos.x = pos_x;
                value = read_imagef(correction_in, nearest_sampler, read_pos).x;
                value.x -= values[median_offset];
                write_imagef(correction_out, (int2)(pos_x, pos_y), value);
                }
            }
        }
    """
