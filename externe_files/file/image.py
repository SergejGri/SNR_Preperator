''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

from externe_files.file.image_formats import *
from externe_files.file.info import *
from externe_files.image.cropping import *

#from file.image_formats import *
#from file.info import *
#from image.cropping import *

def load(filename, crops=((0, 0), (0, 0)), raw_shape=None, raw_header=0, raw_dtype='f4', output_dtype=None, **kwargs):
    '''
    funcion for reading images (2D arrays)

    :param filename:        str with filename
    :param ycrop:           crop parameters in y direction, axis = 0
    :param xcrop:           crop parameters in x direction, axis = 1
    only for raw images:
    :param raw_shape:       raw shape (y, x)
    :param raw_header:      header in bytes
    :param raw_dtype:       dtype in numpy string notation (e.g. 'f4' for float32 and '<u2' for uint16)
    :param output_dtype:    dtype for output, keeps file dtype by default
    :return:
    '''
    suffix = separate_suffix(filename)[1]
    if raw_shape is not None:
        if len(raw_shape) == 3:
            raw_shape = raw_shape[1:]
        assert separate_suffix(filename)[1] not in IMAGE_FORMATS, 'binary image format used with raw load'
        with open(filename, mode='r+b') as file:
            dtype_nbytes = np.dtype(raw_dtype).itemsize
            file.seek(raw_header + dtype_nbytes*crops[0][0]*raw_shape[1])
            read_shape = get_cropped_shape(raw_shape, (crops[0], None))
            try:
                image = np.fromfile(file, dtype=raw_dtype, count=(read_shape[0]*read_shape[1])).reshape(read_shape)
                if output_dtype is not None:
                    image = image.astype(output_dtype)
                image = crop_arr(image, (None, crops[1]))
                return image
            except ValueError:
                pass
    else:
        if suffix in ('raw', ''):
            settings = load_info(filename, optional=False)
            return load(filename, crops=crops, raw_shape=settings['shape'][::-1],
                        raw_dtype=settings['dtype'], output_dtype=output_dtype)
        else:
            pil_image = PIL.Image.open(filename)
            pil_image = crop_pil_image(pil_image, crops, pil_image.size[::-1])
            image = np.array(pil_image)
            if output_dtype is not None:
                image = image.astype(output_dtype)
            return image


def save(image: np.ndarray, filename, suffix=None, index=None, output_dtype=None, unchanged_filename=True,
         png_compress_level=4):
    ''' function for saving images (2D arrays)

    the folders required to save the image will be generated automatically

    :param image:           array with image
    :param filename:        filename string
    :param output_dtype:    output dtype
    :return:
    '''
    try:
        _save(image, filename, suffix, index, output_dtype, unchanged_filename, png_compress_level)
    except OSError:
        os.makedirs(os.path.dirname(filename))
        _save(image, filename, suffix, index, output_dtype, unchanged_filename, png_compress_level)

def _save(image: np.ndarray, filename, suffix=None, index=None, output_dtype=None, unchanged_filename=True,
         png_compress_level=4):
    ''' function for saving images (2D arrays)

    :param image:           array with image
    :param filename:        filename string
    :param output_dtype:    output dtype
    :return:
    '''
    if index is not None:
        index_text = '_{:04}'.format(index)
    else: index_text = ''
    if suffix is None:
        filename_, suffix = separate_suffix(filename)
    else:
        filename_ = filename
    if output_dtype is not None:
        image = image.astype(output_dtype)
    else:
        output_dtype = image.dtype
    if suffix == 'raw':
        if not unchanged_filename: filename = filename_+'{}_{}_{}.raw'.format(index_text, image.shape, output_dtype)
        image.tofile(filename)
        save_info(filename, image.shape, output_dtype)
    else:
        if suffix == '':
            filename += '.tif'
        if not unchanged_filename and index is not None:
            filename = filename_ + index_text + '.' + suffix
        pil_args=dict()
        if filename[-3:] == 'png':
            pil_args['compress_level'] = png_compress_level
        PIL.Image.fromarray(image).save(filename, **pil_args)
    return filename


def save_measurement(image, filename, suffix=None, index=None, output_dtype=None, unchanged_filename=False,
                     png_compress_level=4):
    if os.path.isfile(filename):
        raise FileExistsError(f'image already exists, choose a different filename ({filename})')
    fname = save(image, filename, suffix, index, output_dtype, unchanged_filename, png_compress_level)
    #print(f'saved image to {fname}')


def crop_pil_image(pil_image, crops, imshape):
    right = crops[1][1] if crops[1][1] > 0 else imshape[1] + crops[1][1]
    lower = crops[0][1] if crops[0][1] > 0 else imshape[0] + crops[0][1]
    return pil_image.crop((crops[1][0], crops[0][0], right, lower))


def scale_to_dtype(arr, input_val_range=None, new_dtype=np.uint16):
    if np.dtype(new_dtype) in (np.uint16, np.uint8):
        #print('rescaling from', arr.min(), arr.max(), input_val_range, new_dtype)
        if input_val_range is None:
            input_val_range = arr.min(), arr.max()
        if np.dtype(new_dtype) == np.uint16:  max_val = 2**16-1
        elif np.dtype(new_dtype) == np.uint8: max_val = 2**8-1
        diff = input_val_range[1] - input_val_range[0]
        arr = arr.astype("f4", copy=True)
        arr -= input_val_range[0]
        arr *= max_val / diff
        arr = np.clip(arr, 0, max_val).astype(new_dtype, copy=False)
    # float32 is not changed
    return arr


def unscale_from_dtype(arr, input_val_range, old_dtype=None):
    if old_dtype is None:
        old_dtype = arr.dtype
    if "u" in np.dtype(old_dtype).str:
        #print('unscaling from', arr.min(), arr.max(), input_val_range, old_dtype)
        if "2" in np.dtype(old_dtype).str:  max_val = 2**16-1
        elif "1" in np.dtype(old_dtype).str: max_val = 2**8-1
        elif "4" in np.dtype(old_dtype).str: max_val = 4**8-1
        diff = input_val_range[1] - input_val_range[0]
        arr = arr.astype('f4')
        arr *= diff / max_val
        arr += input_val_range[0]
        #print('unscaled to', arr.min(), arr.max(), arr.write_dtype)
    # float32 is not changed
    return arr