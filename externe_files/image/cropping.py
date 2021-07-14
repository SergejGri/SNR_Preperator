''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import numpy as np, traceback
import image


def newaxs(keep_ax, dim):
    return tuple(None if ax != keep_ax else slice(None) for ax in range(dim))


def get_valid_crops(crops, shape):
    new_crops = []
    for crop, length in zip(crops, shape):
        if crop is None:
            new_crops.append((0, length))
        else:
            new_crops.append((np.clip(crop[0], 0, length), np.clip(crop[1], -length, length)))
    return new_crops


def get_valid_crops_len(crop, ax_length, bin_factor=1):
    if bin_factor > 1:
        crop = crop[0]*bin_factor, crop[1]*bin_factor
    crop_out = np.clip(crop[0], 0, ax_length), np.clip(crop[1], -ax_length, ax_length)
    #print('get_valid_crops_len', crop, crop_out, ax_length, bin_factor)
    return crop_out


def crop_list(list, crop=(0, 0)):
    if crop[1] == 0: crop = crop[0], None
    return list[crop[0]:crop[1]]


def crop_arr(arr, crops):
    return arr[tuple((slice(crop[0], (crop[1] if crop[1] != 0 else None)) if crop is not None else slice(None)) for crop in crops)]


def get_slice(crop):
    if crop[1] != 0:
        return slice(*crop)
    else:
        return slice(crop[0], None)


def get_cropped_shape(shape, crops):
    new_shape = []
    for length, crop in zip(shape, crops):
        if crop is not None:
            if crop[1] <= 0:
                new_shape.append(length-crop[0]+crop[1])
            else:
                new_shape.append(crop[1]-crop[0])
        else:
            new_shape.append(length)
    return new_shape


def check_crops(shape, crops):
    assert np.all(np.array(get_cropped_shape(shape, crops)) > 0), 'cropping is larger than axis length'
    for crop in crops:
        if crop[1] > 0:
            assert crop[0] < crop[1], 'negative crop range'
        if not crop[0] == crop[1] == 0:
            assert crop[0] != crop[1], 'empty crop result'


def get_positive_crops(shape, crops):
    new_crops = []
    for length, crop in zip(shape, crops):
        if crop is None:
            new_crops.append((0, length))
        elif crop[1] > 0:
            new_crops.append(crop)
        else:
            new_crops.append((crop[0], length+crop[1]))
    return new_crops


def get_crops(shape_large, shape_small, center=None, corner=None):
    shape_large_arr, shape_small_arr = np.array(shape_large, dtype=np.int64), np.array(shape_small, dtype=np.int64)
    assert np.all(np.greater_equal(shape_large_arr, shape_small_arr)), 'shape_large must be larger than shape_small'
    assert center is not None or corner is not None
    if corner is None:
        center_arr = np.array(center, dtype=np.int64)
        half_diffs = (shape_large_arr - shape_small_arr) // 2
        assert np.all(np.less_equal(center_arr, half_diffs)), 'center must be smaller than shape diffs/2'
        lower_indices = half_diffs + center_arr
    else:
        diffs = (shape_large_arr - shape_small_arr)
        assert np.all(np.less_equal(np.array(corner), diffs)), 'corner must be smaller than shape diffs'
        lower_indices = np.array(corner)
    upper_indices = lower_indices + shape_small_arr
    return [(lower_indices[k], upper_indices[k]) for k in range(len(shape_large))]


def get_center_range(shape_large, shape_small):
    shape_large_arr, shape_small_arr = np.array(shape_large, dtype=np.int64), np.array(shape_small, dtype=np.int64)
    assert np.all(np.greater_equal(shape_large_arr, shape_small_arr)), 'shape_large must be larger than shape_small'
    return (shape_large_arr - shape_small_arr) // 2


def get_identical_crops_for_padding(length):
    return [(length, -length) for k in range(3)]


def get_slice_subrange(index, ranges_list, additional_range=0, minimal_range=0):
    all_filters_pad_range = image.fourier.sum_pad_ranges(ranges_list)
    if index+1 < len(ranges_list):
        next_filters_pad_range = image.fourier.sum_pad_ranges(ranges_list[index:])
    else:
        next_filters_pad_range = 0
    crop_length = int(all_filters_pad_range - max((next_filters_pad_range + additional_range), minimal_range))
    if crop_length > 0 and crop_length < all_filters_pad_range:
        return slice(crop_length, -crop_length), slice(crop_length, -crop_length), slice(crop_length, -crop_length)
    else:
        return slice(None, None), slice(None, None), slice(None, None)


keys = 'zcrop', 'ycrop', 'xcrop'
def convert_from_old_crop_format(crops):
    if type(crops) is dict:
        print('WARNING, found old crop format\n', traceback.print_stack())
        new_crops = []
        for key in keys:
            try:                new_crops.append(crops[key])
            except KeyError:    pass
        return tuple(new_crops)
    else:
        return crops