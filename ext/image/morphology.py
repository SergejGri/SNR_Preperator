''' volume image morphology functions
written by Maximilian Ullherr, maximilian.ullherr@physik.uni-wuerzburg.de, Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

License for this code:
Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

import numpy as np, gc
from scipy import ndimage

verbose = False

def subdivide_objects(mask, dt, throat_step_size=1, min_object_diff=1, min_object_fract=1.2,
                      min_marker_size=5, iterations=50, dt_scale=1):
    div_step_vals = mask.astype('u1')  # contains marker classes,
    # the number signifies the iteration step at which the final subdivision happened
    statistics_text = ''

    iter_index = 0; iteration_stable = False
    while iter_index < iterations + 1:
        iter_index += 1
        iter_labels = (div_step_vals == iter_index).astype('i4')  # this is the mask, for in-place
        mask_mean = np.mean(iter_labels)
        iter_number_objects = ndimage.label(iter_labels, output=iter_labels)
        gc.collect()

        iter_view_slices = ndimage.find_objects(iter_labels)
        if iter_index < 10:
            min_throat_radius = (throat_step_size * iter_index) * dt_scale
        else:
            min_throat_radius = (throat_step_size * iter_index * np.exp((iter_index - 10) / 20)) * dt_scale
        min_object_radius = min_throat_radius * min_object_fract + min_object_diff * dt_scale

        text = 'iter: {} statistics, mask fraction: {:.5g}, min_throat_radius: {:.2f}; min_object_radius: {:.2f}'.format(
            iter_index, mask_mean, min_throat_radius / dt_scale, min_object_radius / dt_scale)
        if verbose: print(text)
        statistics_text += text + '\n'

        has_subdiv = 0
        for index, view_slice in enumerate(iter_view_slices):
            labels_view = iter_labels[view_slice]
            object_mask = (labels_view == (index + 1))

            local_subdiv = subdivide_one_object(dt[view_slice], object_mask, iter_index + 1,
                                                min_throat_radius, min_object_radius,
                                                min_marker_size)

            if local_subdiv is not False:
                div_step_vals[view_slice][object_mask] = local_subdiv[object_mask]
                has_subdiv += 1

            del local_subdiv, object_mask, labels_view
            if index % 1000 == 0:
                gc.collect()

        del iter_labels; gc.collect()

        mask = (div_step_vals == iter_index).astype('i4')
        p_number1 = ndimage.label(mask, output=mask)
        del mask
        if has_subdiv:
            mask = (div_step_vals == (iter_index + 1)).astype('i4')
            p_number2 = ndimage.label(mask, output=mask)
            del mask
        else:
            p_number2 = 0

        text = 'number of particles, original: {}, kept: {}, subdivided: {} into: {}'.format(
            iter_number_objects, p_number1, has_subdiv, p_number2)
        if verbose: print(text)
        statistics_text += text + '\n'

        if not has_subdiv:
            iteration_stable = True
            break

    labels = (div_step_vals > 0).astype('i4')
    number_objects = ndimage.label(labels, output=labels)
    text = ('final number of objects: {} {}'.format(number_objects, ('(with iteration stable)' if iteration_stable else '')))
    if verbose: print(text)
    statistics_text += text + '\n'
    return labels, statistics_text, number_objects


def subdivide_one_object(local_dt, local_mask, next_iter_index, min_throat_radius, min_object_radius, min_marker_size):
    div_mask = (local_dt * local_mask) > min_throat_radius
    if np.sum(div_mask) == 0:
        return False
    div_labels, number_div = ndimage.label(div_mask)
    div_slices = ndimage.find_objects(div_labels)

    local_subdiv = np.zeros_like(local_mask, dtype='u1')
    has_valid_divisions = 0

    for index, div_slice in enumerate(div_slices):
        div_object_mask = (div_labels[div_slice] == (index + 1))

        shrinked_sum = np.sum(local_dt[div_slice][div_object_mask] > min_object_radius)
        if shrinked_sum > min_marker_size:
            has_valid_divisions += 1
            local_subdiv[div_slice][div_object_mask] = next_iter_index
            #if verbose: print('local_subdiv[div_slice][div_object_mask].shape', local_subdiv[div_slice][div_object_mask].shape)

    if has_valid_divisions > 0:  # return subdivision result
        return local_subdiv
    else:  # protect small particles
        return False