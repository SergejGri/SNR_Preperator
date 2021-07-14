''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import numpy as np


def bin_down_1d_arr(arr: np.ndarray, scale_factor: int):
    if scale_factor == 1:
        return arr
    else:
        downscaled = None
        stop = arr.shape[0] - arr.shape[0] % scale_factor
        for start in range(scale_factor):
            if downscaled is None:
                downscaled = np.copy(arr[start:stop:scale_factor])
            else:
                downscaled += arr[start:stop:scale_factor]
        downscaled /= scale_factor
        return downscaled.astype(arr.dtype)


def bin_down_image(image: np.ndarray, scale_factor: int):
    if scale_factor == 1:
        return image
    else:
        downscaled = None
        stops = np.asarray(image.shape) - np.asarray(image.shape) % scale_factor
        for y_start in range(scale_factor):
            for x_start in range(scale_factor):
                if downscaled is None:
                    downscaled = image[y_start:stops[0]:scale_factor, x_start:stops[1]:scale_factor].astype('f4', copy=True)
                else:
                    downscaled += image[y_start:stops[0]:scale_factor, x_start:stops[1]:scale_factor]
        downscaled /= np.float64(scale_factor**2)
        return downscaled.astype(image.dtype)


def bin_down_volume(volume: np.ndarray, scale_factor: int):
    if scale_factor == 1:
        return volume
    else:
        downscaled = None
        stops = np.asarray(volume.shape) - np.asarray(volume.shape) % scale_factor
        for z_start in range(scale_factor):
            for y_start in range(scale_factor):
                for x_start in range(scale_factor):
                    if downscaled is None:
                        downscaled = volume[z_start:stops[0]:scale_factor, y_start:stops[1]:scale_factor, x_start:stops[2]:scale_factor].astype('f4', copy=True)
                    else:
                        downscaled += volume[z_start:stops[0]:scale_factor, y_start:stops[1]:scale_factor, x_start:stops[2]:scale_factor]
        downscaled /= scale_factor**3
        return downscaled.astype(volume.dtype)


def imscale(im, fact=1):
    if np.isclose(fact, 1):
        return im
    elif fact < 1:
        scale_fact = int(1/fact)
        return bin_down_image(im, scale_fact)
    else:
        scale_fact = int(fact)
        return np.repeat(np.repeat(im, scale_fact, axis=0), scale_fact, axis=1)


def _assign_scaled_down_view(target, source, scale_factor):
    r0 = target.shape[0] - target.shape[0] % scale_factor
    r1 = target.shape[1] - target.shape[1] % scale_factor
    if scale_factor == 2:
        target[:r0:2, :r1:2], target[:r0:2, 1:r1:2], target[1:r0:2, :r1:2], target[1:r0:2, 1:r1:2] = 4*(source,)
    elif scale_factor == 3:
        (target[:r0:3, :r1:3], target[:r0:3, 1:r1:3], target[:r0:3, 2:r1:3], 
         target[1:r0:3, :r1:3], target[1:r0:3, 1:r1:3],  target[1:r0:3, 2:r1:3], 
         target[2:r0:3, :r1:3], target[2:r0:3, 1:r1:3], target[2:r0:3, 2:r1:3]) = 9*(source,)
    elif scale_factor == 4:
        (target[:r0:4, :r1:4], target[:r0:4, 1:r1:4], target[:r0:4, 2:r1:4], target[:r0:4, 3:r1:4],
         target[1:r0:4, :r1:4], target[1:r0:4, 1:r1:4], target[1:r0:4, 2:r1:4], target[1:r0:4, 3:r1:4],
         target[2:r0:4, :r1:4], target[2:r0:4, 1:r1:4], target[2:r0:4, 2:r1:4], target[2:r0:4, 3:r1:4],
         target[3:r0:4, :r1:4], target[3:r0:4, 1:r1:4], target[3:r0:4, 2:r1:4], target[3:r0:4, 3:r1:4]) = 16*(source,)
