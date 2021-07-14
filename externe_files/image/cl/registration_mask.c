//Copyright 2015-2020 University Wuerzburg.

//Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
//1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

//THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
constant sampler_t linear_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_LINEAR;
constant sampler_t nearest_sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

float min_distance_to_01(float val){
    return fmin(val, 1.f-val);
}

float2 constant_noise_diff_shifts(float2 shift_vec){
    float2 shift_add;
    float a, d, b;

    float2 shift_dist = shift_vec - floor(shift_vec);
    const float2 C = shift_dist*shift_dist + (1.f - shift_dist)*(1.f - shift_dist);
    if (min_distance_to_01(shift_dist.y) > min_distance_to_01(shift_dist.x)){
        b = 1.f + fabs(1.f - 2.f*shift_dist.x)*C.y;
        d = b*b - 2.f*(1.f+C.y)*C.y*C.x;
        a = (b - sqrt(d))/(2.f*(1.f + C.y));
        if (shift_dist.x < 0.5f){
            shift_add = (float2)(a, 0.f);
            }
        else{
            shift_add = (float2)(1.f-a, 0.f);
            }
        }
    else{
        b = 1.f + fabs(1.f - 2.f*shift_dist.y)*C.x;
        d = b*b - 2.f*(1.f+C.x)*C.x*C.y;
        a = (b - sqrt(d))/(2.f*(1.f + C.x));
        if (shift_dist.y < 0.5f){
            shift_add = (float2)(0.f, a);
            }
        else{
            shift_add = (float2)(0.f, 1.f-a);
            }
        }
    return(shift_add);
    //float2 vec = (float2)(0.f, 0.f);
    //return(vec);
}

void kernel shifted_params_square_diff_2d_subpixel(read_only image2d_t image1, read_only image2d_t image2,
                                              read_only image2d_t mask1, read_only image2d_t mask2,
                                              global float* result_values, global float* result_weights,
                                              int3 result_shape, int block_size,
                                              global float2* shift_vecs){
    const int3 result_pos = (int3)(get_global_id(2), get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 block_origin = block_size*result_pos.xy;
    const int2 shape = get_image_dim(image1);
    const int2 block_end = min(block_size+block_origin, shape)-1; // because of shift_add

    float2 read_pos_1, read_pos_2;
    float2 shift_vec = shift_vecs[result_pos.z];
    float2 shift_add = constant_noise_diff_shifts(shift_vec);
    shift_vec += shift_add;

    float diff, val1, val2, mval1, mval2, sum_val = 0.f, sum_norm = 0.f;
    float x_norm, y_norm, norm;
    if (all(result_pos < result_shape)){
        for (int k_y = block_origin.y; k_y < block_end.y; k_y++) {
            read_pos_1.y = k_y - shift_add.y + 0.5f;
            read_pos_2.y = k_y - shift_vec.y + 0.5f;
            y_norm = (0.5f < read_pos_2.y) & (read_pos_2.y < shape.y+0.5f);

            for (int k_x = block_origin.x; k_x < block_end.x; k_x++) {
                read_pos_1.x = k_x - shift_add.x + 0.5f;
                read_pos_2.x = k_x - shift_vec.x + 0.5f;
                x_norm = (0.5f < read_pos_2.x) & (read_pos_2.x < shape.x+0.5f);

                mval1 = read_imagef(mask1, linear_sampler, read_pos_1).x;
                mval2 = read_imagef(mask2, linear_sampler, read_pos_2).x;
                val1 = read_imagef(image1, linear_sampler, read_pos_1).x;
                val2 = read_imagef(image2, linear_sampler, read_pos_2).x;

                norm = y_norm*x_norm*(mval1 > 0.9999f)*(mval2 > 0.9999f);
                diff = val1 - val2;

                sum_norm += norm;
                sum_val += norm*diff*diff;
            }
        }

        int write_index = result_shape.x*(result_shape.y*result_pos.z + result_pos.y) + result_pos.x;
        result_values[write_index] = sum_val;
        result_weights[write_index] = sum_norm;
    }
}

void kernel shifted_params_square_diff_2d(read_only image2d_t image1, read_only image2d_t image2,
                                              read_only image2d_t mask1, read_only image2d_t mask2,
                                              global float* result_values, global float* result_weights,
                                              int3 result_shape, int block_size,
                                              global float2* shift_vecs){
    const int3 result_pos = (int3)(get_global_id(2), get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 block_origin = block_size*result_pos.xy+1;
    const int2 shape = get_image_dim(image1);
    const int2 block_end = min(block_size+block_origin, shape)-1; // because of shift_add

    int2 read_pos_1, read_pos_2;
    int2 shift_vec = convert_int2(shift_vecs[result_pos.z]);

    float diff, val1, val2, mval1, mval2, sum_val = 0.f, sum_norm = 0.f;
    float x_norm, y_norm, norm;
    if (all(result_pos < result_shape)){
        for (int k_y = block_origin.y; k_y < block_end.y; k_y++) {
            read_pos_1.y = k_y;
            read_pos_2.y = k_y - shift_vec.y;
            y_norm = (0 < read_pos_2.y) & (read_pos_2.y < shape.y);

            for (int k_x = block_origin.x; k_x < block_end.x; k_x++) {
                read_pos_1.x = k_x;
                read_pos_2.x = k_x - shift_vec.x;
                x_norm = (0 < read_pos_2.x) & (read_pos_2.x < shape.x);

                mval1 = read_imagef(mask1, nearest_sampler, read_pos_1).x;
                mval2 = read_imagef(mask2, nearest_sampler, read_pos_2).x;
                val1 = read_imagef(image1, nearest_sampler, read_pos_1).x;
                val2 = read_imagef(image2, nearest_sampler, read_pos_2).x;

                norm = y_norm*x_norm*(mval1 > 0.9999f)*(mval2 > 0.9999f);
                diff = val1 - val2;

                sum_norm += norm;
                sum_val += norm*diff*diff;
            }
        }

        int write_index = result_shape.x*(result_shape.y*result_pos.z + result_pos.y) + result_pos.x;
        result_values[write_index] = sum_val;
        result_weights[write_index] = sum_norm;
    }
}


void kernel shiftscaled_params_square_diff_2d(read_only image2d_t image1, read_only image2d_t image2,
                                              read_only image2d_t mask1, read_only image2d_t mask2,
                                              global float* result_values, global float* result_weights,
                                              int3 result_shape, int block_size,
                                              global float2* shift_vecs,
                                              float2 scale_origin, float scale_factor){
    const int3 result_pos = (int3)(get_global_id(2), get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 block_origin = block_size*result_pos.xy;
    const int2 shape = get_image_dim(image1);
    const int2 block_end = min(block_size+block_origin, shape);
    float2 read_pos;
    int2 pos_i;

    const float2 shift_vec = shift_vecs[result_pos.z];

    float diff, val1, val2, mval1, mval2, sum_val = 0.f, sum_norm = 0.f;
    float x_norm, y_norm, norm;
    if (all(result_pos < result_shape)){
        for (int k_y = block_origin.y; k_y < block_end.y; k_y++) {
            pos_i.y = k_y;
            read_pos.y = (k_y - scale_origin.y)*scale_factor + scale_origin.y - shift_vec.y + 0.5f;
            y_norm = (0.5f <= read_pos.y) & (read_pos.y <= shape.y-0.5f);

            for (int k_x = block_origin.x; k_x < block_end.x; k_x++) {
                pos_i.x = k_x;
                read_pos.x = (k_x - scale_origin.x)*scale_factor + scale_origin.x - shift_vec.x + 0.5f;
                x_norm = (0.5f <= read_pos.x) & (read_pos.x <= shape.x-0.5f);

                mval1 = read_imagef(mask1, nearest_sampler, pos_i).x;
                mval2 = read_imagef(mask2, linear_sampler, read_pos).x;
                val1 = read_imagef(image1, nearest_sampler, pos_i).x;
                val2 = read_imagef(image2, linear_sampler, read_pos).x;

                norm = y_norm*x_norm*(mval1 > 0.9999f)*(mval2 > 0.9999f);
                diff = val1 - val2;

                sum_norm += norm;
                sum_val += norm*diff*diff;
            }
        }

        int write_index = result_shape.x*(result_shape.y*result_pos.z + result_pos.y) + result_pos.x;
        result_values[write_index] = sum_val;
        result_weights[write_index] = sum_norm;
    }
}

void kernel shiftrot_params_square_diff_2d(read_only image2d_t image1, read_only image2d_t image2,
                                              read_only image2d_t mask1, read_only image2d_t mask2,
                                          global float* result_values, global float* result_weights,
                                          int3 result_shape, int block_size,
                                          global float2* shift_vecs,
                                          float2 rot_origin, float rot_angle){
    const int3 result_pos = (int3)(get_global_id(2), get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 block_origin = block_size*result_pos.xy;
    const int2 shape = get_image_dim(image1);
    const int2 block_end = min(block_size+block_origin, shape);
    float2 read_pos, read_pos_rot;
    int2 pos_i;

    float cos_val = cos(-rot_angle), sin_val = sin(-rot_angle);

    const float2 shift_vec = shift_vecs[result_pos.z];

    float diff, val1, val2, mval1, mval2, sum_val = 0.f, sum_norm = 0.f;
    float x_norm, y_norm, norm;
    if (all(result_pos < result_shape)){
        for (int k_y = block_origin.y; k_y < block_end.y; k_y++) {
            pos_i.y = k_y;
            read_pos.y = k_y + shift_vec.y - rot_origin.y + 0.5f;
            y_norm = (0.5f <= read_pos.y) & (read_pos.y <= shape.y-0.5f);

            for (int k_x = block_origin.x; k_x < block_end.x; k_x++) {
                pos_i.x = k_x;
                read_pos.x = k_x + shift_vec.x - rot_origin.x + 0.5f;
                x_norm = (0.5f <= read_pos.x) & (read_pos.x <= shape.x-0.5f);

                read_pos_rot.x = cos_val*read_pos.x - sin_val*read_pos.y;
                read_pos_rot.y = sin_val*read_pos.x + cos_val*read_pos.y;
                read_pos_rot += rot_origin;

                mval1 = read_imagef(mask1, nearest_sampler, pos_i).x;
                mval2 = read_imagef(mask2, linear_sampler, read_pos).x;
                val1 = read_imagef(image1, nearest_sampler, pos_i).x;
                val2 = read_imagef(image2, linear_sampler, read_pos).x;


                norm = y_norm*x_norm*(mval1 > 0.9999f)*(mval2 > 0.9999f);
                diff = val1 - val2;

                sum_norm += norm;
                sum_val += norm*diff*diff;
            }
        }

        int write_index = result_shape.x*(result_shape.y*result_pos.z + result_pos.y) + result_pos.x;
        result_values[write_index] = sum_val;
        result_weights[write_index] = sum_norm;
    }
}

void kernel shift_image(read_only image2d_t image, write_only image2d_t result_image,
                             float2 shift_vec){
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 shape = get_image_dim(result_image);

    if (all(pos < shape)) {
        float2 read_pos = convert_float2(pos) + shift_vec + 0.5f;
        float val = read_imagef(image, linear_sampler, read_pos).x;
        write_imagef(result_image, pos, (float4)(val, 0, 0, 0));
    }
}

void kernel shiftscale_image_inv(read_only image2d_t image, write_only image2d_t result_image,
                             float2 shift_vec, float2 scale_origin, float scale_factor){
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 shape = get_image_dim(result_image);

    if (all(pos < shape)) {
        float2 read_pos = (convert_float2(pos) - scale_origin)*scale_factor + scale_origin - shift_vec + 0.5f;
        float val = read_imagef(image, linear_sampler, read_pos).x;
        write_imagef(result_image, pos, (float4)(val, 0, 0, 0));
    }
}
void kernel shiftscale_image(read_only image2d_t image, write_only image2d_t result_image,
                             float2 shift_vec, float2 scale_origin, float scale_factor){
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 shape = get_image_dim(result_image);

    if (all(pos < shape)) {
        float2 read_pos = native_divide((convert_float2(pos) + shift_vec - scale_origin), scale_factor) + scale_origin + 0.5f;
        float val = read_imagef(image, linear_sampler, read_pos).x;
        write_imagef(result_image, pos, (float4)(val, 0, 0, 0));
    }
}

void kernel shiftrot_image_inv(read_only image2d_t image, write_only image2d_t result_image,
                           float2 shift_vec, float2 rot_origin, float rot_angle){
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 shape = get_image_dim(result_image);
    if (all(pos < shape)) {
        float cos_val = cos(-rot_angle), sin_val = sin(-rot_angle);
        float2 read_pos_rot;
        float2 read_pos = convert_float2(pos) + shift_vec + 0.5f;
        read_pos -= rot_origin;
        read_pos_rot.x = cos_val*read_pos.x - sin_val*read_pos.y;
        read_pos_rot.y = sin_val*read_pos.x + cos_val*read_pos.y;
        read_pos_rot += rot_origin;

        float val = read_imagef(image, linear_sampler, read_pos_rot).x;
        write_imagef(result_image, pos, (float4)(val, 0, 0, 0));
    }
}

void kernel shiftrot_image(read_only image2d_t image, write_only image2d_t result_image,
                               float2 shift_vec, float2 rot_origin, float rot_angle){
    const int2 pos = (int2)(get_global_id(1), get_global_id(0)); // work size is in numpy order = yx
    const int2 shape = get_image_dim(result_image);
    if (all(pos < shape)) {
        float cos_val = cos(rot_angle), sin_val = sin(rot_angle);
        float2 read_pos_rot;
        float2 read_pos = convert_float2(pos) + 0.5f;
        read_pos -= rot_origin;
        read_pos_rot.x = cos_val*read_pos.x - sin_val*read_pos.y;
        read_pos_rot.y = sin_val*read_pos.x + cos_val*read_pos.y;
        read_pos_rot += rot_origin - shift_vec;

        float val = read_imagef(image, linear_sampler, read_pos_rot).x;
        write_imagef(result_image, pos, (float4)(val, 0, 0, 0));
    }
}
