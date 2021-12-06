''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import json
from ext.file.path import *
#from file.path import *


class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if np.iterable(o) or type(o) is np.ndarray:
            return list(o)
        elif issubclass(o.__class__, (dict,)):
            return dict(**o)
        else:
            try: # conversion from numpy numbers
                kind = o.dtype.kind
                if kind in ('i', 'u'):
                    return int(o)
                elif kind in ('S', 'U'):
                    return str(o)
                else:
                    return float(o)
            except (ValueError, TypeError):
                return json.JSONEncoder.default(self, o)


def get_info_filename(filename):
    if type(filename) in (tuple, list):
        assert len(filename[0]) > 1
        filename = os.path.commonprefix(filename)
        filename = filename.rstrip("_0")
        #print("found info file name", filename)
    else:
        assert type(filename) is str, "{}".format(filename)
    if filename == "":
        filename = "info"
    return modify_path(filename, suffix='info')


def load_info(image_filename, optional=True):
    info_filename = get_info_filename(image_filename)
    #print('load info from', info_filename, image_filename)
    if os.path.isfile(info_filename):
        try:
            with open(info_filename, 'r') as fp:
                info = json.load(fp)
            #print('info read', info)
            return info
        except Exception:
            #print(f'WARNING: invalid info file at {info_filename}')
            return None

    elif not optional:
        raise OSError(f'no info file exists at {info_filename}')


def save_info(image_filename, shape, dtype, value_range=None, **info):
    json_str = '{'
    json_str += f'"shape": {list(shape[::-1])},\n'
    if value_range is not None:
        json_str += '"value_range": {},\n'.format(list(value_range))
    json_str += f'"dtype": "{np.dtype(dtype).str}"\n'
    if len(info) > 0:
        try:
            encoded = ExtendedJSONEncoder(sort_keys=True, skipkeys=True, indent=2).encode(info)[1:]
            json_str += ','
            json_str += encoded
        except TypeError as err:
            print('WARNING: adding other info to info file failed:', err)
            json_str += '}'
    else:
        json_str += '}'

    info_filename = get_info_filename(image_filename)
    with open(info_filename, 'w') as fp:
        fp.write(json_str)

    #print(f'info written {json_str}')
    #with open(info_filename, 'r') as fp:
    #    print('reloaded', json.load(fp))
