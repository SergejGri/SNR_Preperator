//Copyright 2015-2020 University Wuerzburg.

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant sampler_t int_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;

kernel void median_filter(read_only image2d_t image, write_only image2d_t result,
                          constant char2* offsets, int num_offsets){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    float values[256];
    num_offsets = min(num_offsets, 256);
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(image);
    if (all(pos < shape)){
        for (int k=0; k < num_offsets; k++){
            values[k] = read_imagef(image, sampler, pos+convert_int2(offsets[k])).x;
            }
        write_imagef(result, pos, (float4)(quickselect_median(values, num_offsets), 0, 0, 0));
        }
    }
kernel void conditioned_median_filter(read_only image2d_t image, write_only image2d_t result,
                          constant char2* offsets, int num_offsets, float threshold){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    float values[256];
    num_offsets = min(num_offsets, 256);
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(image);
    if (all(pos < shape)){
        float pos_val = read_imagef(image, sampler, pos).x;
        for (int k=0; k < num_offsets; k++){
            values[k] = read_imagef(image, sampler, pos+convert_int2(offsets[k])).x;
            }
        float median_val = quickselect_median(values, num_offsets);
        if (fdim(pos_val, median_val) > threshold){
            write_imagef(result, pos, (float4)(median_val, 0, 0, 0));
            }
        else{
            write_imagef(result, pos, (float4)(pos_val, 0, 0, 0));
            }
        }
    }

kernel void masked_median_filter(read_only image2d_t image, write_only image2d_t result,
                          constant char2* offsets, int num_offsets,
                          read_only image2d_t mask){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    float values[256];
    num_offsets = min(num_offsets, 256);
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(image);
    if (all(pos < shape)){
        if (read_imagei(mask, sampler, pos).x > 0){
            for (int k=0; k < num_offsets; k++){
                values[k] = read_imagef(image, sampler, pos+convert_int2(offsets[k])).x;
                }
            write_imagef(result, pos, (float4)(quickselect_median(values, num_offsets), 0, 0, 0));
            }
        else{
            write_imagef(result, pos, read_imagef(image, sampler, pos));
            }
        }
    }

kernel void copy_image(read_only image2d_t image, write_only image2d_t result){
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(image);
    if (all(pos < shape)){
        write_imagef(result, pos, read_imagef(image, sampler, pos));
        }
    }

kernel void sum_mask(read_only image2d_t mask, global uint* mask_sum){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(mask);
    if (all(pos < shape)){
        if (read_imagei(mask, sampler, pos).x > 0){
            uint positions_index = atomic_inc(&mask_sum[0]);
            }
        }
    }

kernel void compute_positions(read_only image2d_t mask,
                          global ushort2* positions, global uint* number_positions){
    const int2 pos = (int2)(get_global_id(0), get_global_id(1));
    const int2 shape = get_image_dim(mask);
    if (all(pos < shape)){
        if (read_imagei(mask, sampler, pos).x > 0){
            uint positions_index = atomic_inc(&number_positions[0]);
            positions[positions_index] = convert_ushort2(pos);
            }
        }
    }

kernel void pos_median_filter(read_only image2d_t image, write_only image2d_t result,
                          constant char2* offsets, int num_offsets,
                          global ushort2* positions, global uint* num_positions){
    // offsets are the indices to include in the filter, e.g. [(0, 0), (0, 1), (1, 0), (-1, 0), (0, -1)]
    // positions are the positions for which the filter should be evaluated (similar as ofsets)
    float values[256];
    num_offsets = min(num_offsets, 256);
    const int pos_index = get_global_id(0);
    if (pos_index < num_positions[0]){
        const int2 pos = convert_int2(positions[pos_index]);
        const int2 shape = (int2)(get_image_width(image), get_image_height(image));
        for (int k=0; k < num_offsets; k++){
            values[k] = read_imagef(image, sampler, pos+convert_int2(offsets[k])).x;
            }
        write_imagef(result, pos, (float4)(quickselect_median(values, num_offsets), 0, 0, 0));

        }
    }

