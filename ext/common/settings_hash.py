''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np, hashlib


def partial_dict_copy(input_dict, keys):
    new_dict = dict()
    for key in input_dict:
        if key in keys:
            new_dict[key] = input_dict[key]
    return new_dict


def generate_hash_from_dict(input_dict, excluded_keys=(), included_keys=None, maximum_array_size=10, diagnostic=False):
    # warning: not all entries of array are included for arrays > maximum_array_size [MB]!
    hash_bytes = bytes()
    hash_bytes += add_dict_bytes(input_dict, excluded_keys, included_keys, maximum_array_size)
    if diagnostic:
        generate_hash_from_dict.hash_bytes = hash_bytes
    return hashlib.md5(hash_bytes).hexdigest()


def add_dict_bytes(input_dict, excluded_keys=(), included_keys=None, maximum_array_size=10):
    hash_bytes = bytes()
    if included_keys is None:
        included_keys = input_dict.keys()
    for key in sorted(included_keys):
        if key not in excluded_keys:
            hash_bytes += bytes(key, encoding='utf-8')
            if hasattr(input_dict[key], "keys"):
                hash_bytes += add_dict_bytes(input_dict[key], maximum_array_size=maximum_array_size)
            else:
                try:  # for numpy arrays
                    hash_bytes += stride_to_size(input_dict[key], maximum_array_size).tobytes()
                except AttributeError:
                    if np.iterable(input_dict[key]) and not type(input_dict[key]) is str:
                        hash_bytes += bytes(str(tuple(input_dict[key])), encoding='utf-8')
                    else:
                        hash_bytes += bytes(str(input_dict[key]), encoding='utf-8')
    return hash_bytes


def generate_hash_from_dict_list(dict_list, excluded_keys=(), maximum_array_size=10):
    hash_bytes = bytes()
    for input_dict in dict_list:
        for key in sorted(input_dict.keys()):
            if key not in excluded_keys:
                hash_bytes += bytes(key, encoding='utf-8')
                try:  # for numpy arrays
                    hash_bytes += stride_to_size(input_dict[key], maximum_array_size).tobytes()
                except AttributeError:
                    hash_bytes += bytes(str(input_dict[key]), encoding='utf-8')
    return hashlib.md5(hash_bytes).hexdigest()


def generate_hash_from_list(_list, maximum_array_size=10):
    hash_bytes = bytes()
    for item in _list:
        try:  # for numpy arrays
            hash_bytes += stride_to_size(item, maximum_array_size).tobytes()
        except AttributeError:
            hash_bytes += bytes(str(item), encoding='utf-8')
    return hashlib.md5(hash_bytes).hexdigest()


def strided(arr, stride=1):
    try:
        stride_slice = arr.ndim * (slice(None, None, int(stride)),)
    except TypeError:
        assert len(stride) == arr.ndim; 'stride must be a tuple of length ndim'
        stride_slice = tuple(slice(None, None, s) for s in stride)
    return arr[stride_slice]


def stride_to_size(arr, max_size):
    size = array_size(arr)
    if size > max_size:
        stride = int(np.ceil((size/max_size) ** (1 / arr.ndim)))
        return strided(arr, stride)
    else:
        return arr


def array_size(ndarray, units_1024_exp=2):
    # units_1024_exp is 2 fo MB, 3 for GB, ...
    return np.prod(ndarray.shape)*ndarray.dtype.itemsize/1024**units_1024_exp
