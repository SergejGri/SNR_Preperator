//Copyright 2015-2020 University Wuerzburg.

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


kernel void mask_width_transform(read_only image2d_t mask_in, write_only image2d_t mask_out,
                                 int radius_add){
        const int2 pos = (int2)(0, get_global_id(0));
        const int2 shape = get_image_dim(mask_in);
        if (pos.y < shape.y){
            ushort k_min, k_max, k_width = 0;
            int4 result_value, write_value;
            int mask_val;
            for (ushort k=0; k < shape.x; k++){
                pos.x = k;
                mask_val = read_imagei(mask_in, sampler, pos).x;
                if (mask_val > 0 & k_width < 64){
                    if (k_width == 0){k_min = k;}
                    k_width++;
                    }
                else{
                    if (k_width > 0){
                        k_max = k-1;
                        result_value.x = (int)((float)k_width*1.2f)+radius_add;
                        for (ushort j=k_min; j<=k_max; j++){
                            write_value.x = max(result_value.x, read_imagei(mask_in, sampler, (int2)(j, pos.y)).x);
                            write_imagei(mask_out, (int2)(j, pos.y), write_value);
                            }
                        }
                        k_width = 0;
                    }
                }
            }
        }

kernel void filter_bad_pixels_sino(read_only image2d_t image, write_only image2d_t result,
                          read_only image2d_t mask, int mask_y_pos){
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(image);
    if (all(pos < shape)){
        float values[256];
        int radius = min(read_imagei(mask, sampler, (int2)(pos.x, mask_y_pos)).x, 127);
        if (radius > 0){
            for (int k_radius=1; k_radius <= radius; k_radius++){
                values[k_radius] = read_imagef(image, sampler, pos+(int2)(k_radius, 0)).x;
                values[radius+k_radius] = read_imagef(image, sampler, pos-(int2)(k_radius, 0)).x;
                }
            values[0] = read_imagef(image, sampler, pos+(int2)(radius+1, 0)).x;

            write_imagef(result, pos, (float4)(quickselect_median(values, radius*2+1), 0, 0, 0));
            }
        else{
            write_imagef(result, pos, read_imagef(image, sampler, pos));
            }
        }
    }

kernel void sum_mask1d(read_only image2d_t mask, global uint* mask_sum, int y_pos){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    const int2 pos = (int2)(get_global_id(0), y_pos);
    const int2 shape = (int2)(get_image_width(mask), get_image_height(mask));
    if (all(pos < shape)){
        if (read_imagei(mask, sampler, pos).x > 0){
            uint positions_index = atomic_inc(&mask_sum[0]);
            }
        }
    }
kernel void compute_pos1d(read_only image2d_t mask, int y_pos,
                          global ushort* positions, global uint* number_positions){
    const int2 pos = (int2)(get_global_id(0), y_pos);
    const int2 shape = (int2)(get_image_width(mask), get_image_height(mask));
    if (all(pos < shape)){
        if (read_imagei(mask, sampler, pos).x > 0){
            uint positions_index = atomic_inc(&number_positions[0]);
            positions[positions_index] = pos.x;
            }
        }
    }

kernel void pos1d_median_filter(read_only image2d_t image, write_only image2d_t result,
                          read_only image2d_t mask,
                          global ushort* positions, global uint* num_positions){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    // positions are the positions for which the filter should be evaluated (similar as offsets)
    // the mask image gray value gives the radius of the median filter
    float values[256];
    const int pos_index = get_global_id(0);
    if (pos_index < num_positions[0]){
        int2 offset, offset_0, pos = (int2)(positions[pos_index], get_global_id(1));
        const int2 shape = (int2)(get_image_width(image), get_image_height(image));
        int radius = min(read_imagei(mask, sampler, pos).x, 127);
        if (pos.y < shape.y){
            for (int k_radius=1; k_radius <= radius; k_radius++){
                values[k_radius] = read_imagef(image, sampler, pos+(int2)(k_radius, 0)).x;
                values[radius+k_radius] = read_imagef(image, sampler, pos-(int2)(k_radius, 0)).x;
                }
            values[0] = read_imagef(image, sampler, pos+(int2)(radius+1, 0)).x;
            write_imagef(result, pos, (float4)(quickselect_median(values, 2*radius+1), 0, 0, 0));
            }
        }
    }
