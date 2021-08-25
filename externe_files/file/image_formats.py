''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import io, PIL, numpy as np
import PIL.Image

IMAGE_FORMATS = 'tif', 'jpg', 'png', 'gif', 'jpeg', 'tiff', 'edf', 'bmp'
                # types allowed for binary images if restrict_binary_to_IMAGE_FORMATS (ImagesReader)
IMAGE_IGNORED_TYPES = 'info', 'txt', 'pysettings'
                # file types ignored for all image formats if use_IMAGE_IGNORED_TYPES (ImagesReader)


def file_in_IMAGE_FORMATS(file):
    for suffix in IMAGE_FORMATS:
        if file[-len(suffix):] == suffix:
            return True
    #print('not in IMAGE_FORMATS:', file)
    return False

def file_in_IMAGE_IGNORED_TYPES(file):
    for suffix in IMAGE_IGNORED_TYPES:
        if file[-len(suffix):] == suffix:
            #print('in IMAGE_IGNORED_TYPES:', file)
            return True
    return False



def test_binary_image(filename: str):
    try:
        if filename[-3:] == 'edf':
            with open(filename, 'r', encoding='ascii', errors='replace') as file_io:
                edf_header = ''
                for k in np.arange(20):
                    file_io.seek(k*256)
                    edf_header += file_io.read(256)
                    if k == 0 and edf_header[0] != '{':
                        raise TypeError('not a valid edf header {')

                    if '}' in edf_header[-3:]:
                        break
                    elif k == 19:
                        raise TypeError('not a valid edf header }')

                lines = edf_header.splitlines()
                header_dict = dict()
                for line in lines:
                    try:
                        key, val = line.split('=')
                        key = key.strip(' ')
                        val = val[:-1].strip(' ')
                        header_dict[key]= val
                    except ValueError:
                        pass

                im_dtype = '<' if header_dict['ByteOrder'] == 'LowByteFirst' else '>'
                im_dtype += EDF_DATA_TYPES[header_dict['DataType']]
                im_shape = int(header_dict['Dim_2']), int(header_dict['Dim_1'])
                return True, im_shape, True, im_dtype, int((k+1)*256)

        else:

            with open(filename, mode='rb') as file:
                is_big_endian = file.read(2) == b'MM' # todo: find out where (and if) PIL indicates endianness
                pil_image = PIL.Image.open(file)
                imshape = pil_image.size[1], pil_image.size[0]
                if type(pil_image) == PIL.TiffImagePlugin.TiffImageFile:
                    nheader = pil_image.tag[273][0]
                    number_nbytes = pil_image.tag[258][0]//8

                    if pil_image.mode == 'F':
                        im_dtype = 'f'+str(number_nbytes)
                    elif pil_image.mode[0] == 'I':
                        im_dtype = 'u'+str(number_nbytes)
                    elif pil_image.mode == 'L':
                        im_dtype = 'u1'
                    else:
                        raise TypeError
                    if is_big_endian:
                        im_dtype = '>' + im_dtype
                    else:
                        im_dtype = '<' + im_dtype
                    return True, imshape, pil_image.info['compression']=='raw', im_dtype, nheader
                else:
                    return True, imshape, False, None, 0
    except (OSError, TypeError, ValueError):
        return False, (0, 0), False, None, 0


EDF_DATA_TYPES = {"SignedByte"            : 'u1', "UnsignedByte"          : 'u1',
                  "SignedShort"           : 'i2', "UnsignedShort"         : 'u2',
                  "SignedInteger"         : 'i4', "UnsignedInteger"       : 'u4',
                  "SignedLong"            : 'i8', "UnsignedLong"          : 'u8',
                  "Float"                 : 'f4',
                  "Float32"               : 'f4',
                  "Double"                : 'f8'}



def get_tif_header(imshape, im_dtype, resolution=None):
    with io.BytesIO() as raw:
        pil_args = dict()
        if resolution is not None:
            pil_args['resolution'] = 1/resolution*1e4
            pil_args['x_resolution'] = 1/resolution*1e4
            pil_args['y_resolution'] = 1/resolution*1e4
            pil_args['resolution_unit'] = "cm"

        pil_image_w = PIL.Image.fromarray(np.zeros(imshape, im_dtype), **pil_args)
        pil_image_w.save(raw, format='tiff')

        pil_image_r = PIL.Image.open(raw)
        nheader = pil_image_r.tag[273][0]

        raw.seek(0)
        header = raw.read(nheader)
    return nheader, header