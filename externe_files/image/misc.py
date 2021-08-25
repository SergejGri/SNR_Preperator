''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
from common import *
import numpy as np

def apply_neglog(data, clip=True):
    data = data.astype("f4", copy=False)
    if clip:
        np.fmax(data, 1e-7, out=data)
    np.log(data, out=data)
    data *= -1
    return data

def circular_footprint(radius, dims=2, radius_addition=0.3):
    # function for generating a footprint for a filter with a center pixel (e.g. median filter)
    radius += radius_addition
    coords = np.arange(-np.floor(radius), np.ceil(radius))**2
    if dims == 1:
        return coords <= radius**2
    elif dims == 2:
        return (coords[None, :] + coords[:, None]) <= radius**2
    elif dims == 3:
        return (coords[None, None, :] + coords[:, None, None] + coords[None, :, None]) <= radius**2
    else:
        raise ValueError('wrong ndim: ' + str(dims))

def rounded_box_footprint(side_length, dims=2, radius_addition=0.5):
    # function for generating a footprint for a filter without a center pixel (e.g. binary opening)
    coords = np.linspace(-(side_length-1)/2, (side_length-1)/2, side_length, endpoint=True)**2
    radius = (side_length-1)/2 + radius_addition
    if dims == 2:
        return np.less_equal(coords[None, :] + coords[:, None], radius**2)
    elif dims == 3:
        return np.less_equal(coords[None, None, :] + coords[:, None, None] + coords[None, :, None], radius**2)

