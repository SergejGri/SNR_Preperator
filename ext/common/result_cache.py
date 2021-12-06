''' intermediate results caching functions

use example is in ball_phantom_evaluation.py or ball_phantom_evaluation-simulation.ipynb

intended use from a jupyter notebook:
variant A): call: result_cache.enable_cache('[cache directory]')
            to clear cache: result_cache.clear_cache()
variant B): result_cache.enable_temporary_cache()
            is cleared automatically on a clean interpreter exit (but not when Python crashes)

if no cache is enabled, Data.load() will always raise the NoData exception

use:
from pyEXT import result_cache
result_cache.enable_cache('[cache directory]')

# there can be an arbitrary number of Data objects, but they should have different names (first argument)
cached_data = result_cache.Data('example', param1, param2=param2)  # all parameters/data which influence the result must be args
try:
    # Variant 1:
    arr = cached_data.load()
    # Variant 2:
    arr1, arr2 = cached_data.load()
    # Variant 3:
    arr1, arr2, kwargs = cached_data.load()  # returns kwargs dict here
    param1 = kwargs['param1']

except result_cache.NoData:
    # === compute result here ===
    # Variant 1:
    cached_data.save(data)
    # Variant 2:
    cached_data.save(arr1, arr2)
    # Variant 3:
    cached_data.save(arr1, arr2, param1=param1)  # pass parameters as kwargs here

# when the evaluation is finished
result_cache.clear_cache()

end use

written by Maximilian Ullherr, maximilian.ullherr@physik.uni-wuerzburg.de, Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

License for this code:
Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np, shutil, os, glob, json, atexit, tempfile, hashlib, inspect

__all__ = []  # should not be used with import * (see the two lines below)
version_str = '' # must be overwritten by importing module (caches are only valid for the same evaluation code version)
default_cache_dir = None


class Data:
    def __init__(self, name, *args, function=None, cache_dir=None, **kwargs):
        ''' object representing a set of cached computation results

        WARNING: args and kwargs must include all arguments that influence the result,
        otherwise the there will be false cache hits: the retrieved cached result is incorrect for changed parameters

        Note: caching speed (hash computation) is roughly 1 GB/s on modern hardware for arrays in args/kwargs

        :param name:                short description for the contents
        :param args:                arguments that influence the result (can be ndarray)
        :param function:            if a callable is supplied here, its source code is added to the cache identifier hash
        :param cache_dir:           directory to save cache to
                                    WARNING: if this is None and default_cache_dir is None, no caching will be done!
        :param kwargs:              keyword arguments that influence the result (can be ndarray)
        '''
        self.identifier = name + '_' + self.hash_params(*args, function=function, **kwargs) + '_' + version_str
        if cache_dir is None and default_cache_dir is not None:
            self.cache_dir = default_cache_dir
        else:
            self.cache_dir = cache_dir

    @staticmethod
    def hash_params(*args, function=None, **kwargs):
        '''
        generate a caching hash for identifying a set of input parameters

        use this hash function only for caching of results, not in a security context!

        warning: not all entries of array are included for arrays > maximum_array_size [MB]!
                 it is intended to differentiate between very different input data
                 large arrays where few values are different may get the same hash
                 this behaviour can be disabled by setting maximum_array_size large enough (at the cost of performance)

        :param function:            if a callable is supplied here, its source code is added to the hash
        :param kwargs:              the arguments for the hash
        :return:
        '''
        if function is not None:
            kwargs['function'] = inspect.getsource(function)

        data_hash = hashlib.sha1()
        for arg in args:
            Data.add_obj_to_hash(data_hash, arg)

        Data.add_dict_to_hash(data_hash, kwargs)
        return data_hash.hexdigest()

    @staticmethod
    def add_dict_to_hash(data_hash, kwargs):
        for key in sorted(kwargs.keys()):
            data_hash.update(bytes(key, encoding='utf-8'))
            Data.add_obj_to_hash(data_hash, kwargs[key])

    @staticmethod
    def add_obj_to_hash(data_hash, item):
        if hasattr(item, "keys"):
            Data.add_dict_to_hash(data_hash, item)
        else:
            if issubclass(type(item), np.ndarray):
                if item.nbytes < 1024**2:
                    data_hash.update(item.tobytes())
                else:
                    data_hash.update(item.tobytes())
            else:
                if np.iterable(item) and not type(item) is str:
                    data_hash.update(bytes(str(tuple(item)), encoding='utf-8'))
                else:
                    data_hash.update(bytes(str(item), encoding='utf-8'))

    def save(self, *args, **kwargs):

        ''' save data to the caching directory, to be reloaded with load_data()

        :param args:            ndarrays to save as raw data
        :param kwargs:          parameters to save (item types should be in str, int, float, bool and short lists/dicts
                                with entries of the same type)
                                WARNING: ndarrays passed as kwarg will be saved to json, but restored as list; also this is slow
        :return:
        '''
        self.make_save_cache_dir()
        if self.cache_dir is not None:
            for k, arg in enumerate(args):
                assert issubclass(type(arg), np.ndarray), f'positional arguments must be of type ndarray, is {type(arg)}'
                raw_path = os.path.join(self.cache_dir, self.identifier + f'_{k}.raw')
                save_ndarray(raw_path, arg)

            if len(kwargs) > 0:
                json_str = json_encoder.encode(kwargs)
                save_path = os.path.join(self.cache_dir, self.identifier + '.json')
                with open(save_path, 'w') as file:
                    file.write(json_str)

            print(f'saved cached result to {os.path.join(self.cache_dir, self.identifier)}*')

    def load(self):
        ''' load data that may have been previously saved with save_data()

        kwargs put into save() are returned as a dict (last or only returned entry)

        if no cached data is found, this function simply returns None
        :return:                data saved before,
                                tuple with  (*args, kwargs) from save_data (kwargs entry is dropped if no kwargs were given)
        '''

        self.make_load_cache_dir()
        if self.cache_dir is not None:
            raws_path = os.path.join(self.cache_dir, self.identifier+'_0.raw')
            data = []
            if os.path.isfile(raws_path):
                k = 0
                while True:
                    raw_path = os.path.join(self.cache_dir, self.identifier+f'_{k}.raw')
                    if os.path.isfile(raw_path):
                        data.append(load_ndarray(raw_path)[0])
                        k += 1
                    else:
                        break

            dict_path = os.path.join(self.cache_dir, self.identifier + '.json')
            if os.path.isfile(dict_path):
                with open(dict_path, 'r') as file:
                    data.append(json.load(file))

            if len(data) > 0:
                print(f'loaded cached result from {os.path.join(self.cache_dir, self.identifier)}*')
                if len(data) == 1:
                    return data[0]
                else:
                    return tuple(data)

            else:
                raise NoData
        else:
            raise NoData

    def clear(self):
        ''' delete file for cached data
        :return:
        '''
        self.make_load_cache_dir()
        if self.cache_dir is not None:
            fnames = (os.path.join(self.cache_dir, self.identifier + '.raw'),
                      os.path.join(self.cache_dir, self.identifier + '.json'),
                      os.path.join(self.cache_dir, self.identifier + '_0.raw'))
            for fname in fnames:
                if os.path.isfile(fname):
                    os.remove(fname)

    def make_save_cache_dir(self):
        if self.cache_dir is None and default_cache_dir is not None:
            self.cache_dir = default_cache_dir
        if self.cache_dir is not None:
            if not os.path.isdir(self.cache_dir):
                os.makedirs(self.cache_dir, exist_ok=True)

    def make_load_cache_dir(self):
        if self.cache_dir is None:
            self.cache_dir = default_cache_dir


class NoData(Exception):
    pass


def enable_cache(cache_dir, clear_at_exit=False):
    ''' enable caching and set the default directory to cache_dir


    :param cache_dir:       this directory will be used if none is given to Data
    :param clear_at_exit:   remove cache directory at interpreter exit (not recommended, see clear_cache())
    :return:
    '''
    global default_cache_dir
    default_cache_dir = cache_dir
    print(f'caching of intermediate results enabled, default cache dir:\n{os.path.abspath(default_cache_dir)}')
    if clear_at_exit:
        answer = input(f'this will delete the cache directory "{os.path.abspath(default_cache_dir)}" on interpreter exit, continue? (y/n)')
        if answer == 'y':
            atexit.register(clear_cache, default_cache_dir, ask=False)


def clear_cache(cache_dir=None, ask=True):
    ''' clear the default cache or the given cache_dir


    :param cache_dir:   cache dir to delete, defaults to default_cache_dir
    :param ask:         confirm delete via user input()
    :return:
    '''
    if cache_dir is None:
        if default_cache_dir is None:
            return  # no default dir set, no cleanup necessary
        else:
            cache_dir = default_cache_dir
    if ask:
        answer = input(f'this will delete the cache directory "{os.path.abspath(cache_dir)}", continue? (y/n)')
    else:
        answer = 'y'
    if answer == 'y':
        if os.path.isdir(cache_dir):
            num_files = len(glob.glob(os.path.join(cache_dir, '*')))
            shutil.rmtree(cache_dir)
            print(f'success (caches directory deleted, {num_files} files)')
        else:
            print('success (caches directory does not exist)')
    else:
        print('canceled')


def enable_temporary_cache():
    ''' enable caching to a system-specific temporary directory
    is automatically deleted on a clean interpreter exit

    :return:
    '''
    global default_cache_dir
    default_cache_dir = tempfile.mkdtemp(suffix='_pyETX_results_cache')
    atexit.register(clear_cache, default_cache_dir, ask=False)
    print(f'caching of intermediate results enabled, default cache dir:\n{os.path.abspath(default_cache_dir)}\ncache will be cleared on interpreter exit')


def clear_orphaned_temporary_caches(include_active=False):
    raise NotImplementedError


# ===== helpers =====
class ExtendedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if np.iterable(o) or type(o) is np.ndarray:
            return list(o)
        elif issubclass(o.__class__, (dict,)):
            return dict(**o)
        else:
            try:  # conversion from numpy numbers
                kind = o.dtype.kind
                if kind in ('i', 'u'):
                    return int(o)
                elif kind in ('S', 'U'):
                    return str(o)
                else:
                    return float(o)
            except (ValueError, TypeError):
                return json.JSONEncoder.default(self, o)


def dict_bytes(input_dict, excluded_keys=(), included_keys=None, maximum_array_size=100):
    hash_bytes = bytes()
    if included_keys is None:
        included_keys = input_dict.keys()
    for key in sorted(included_keys):
        if key not in excluded_keys:
            hash_bytes += bytes(key, encoding='utf-8')
            if hasattr(input_dict[key], "keys"):
                hash_bytes += dict_bytes(input_dict[key], maximum_array_size=maximum_array_size)
            else:
                try:  # for numpy arrays
                    hash_bytes += stride_to_size(input_dict[key], maximum_array_size).tobytes()
                except AttributeError:
                    if np.iterable(input_dict[key]) and not type(input_dict[key]) is str:
                        hash_bytes += bytes(str(tuple(input_dict[key])), encoding='utf-8')
                    else:
                        hash_bytes += bytes(str(input_dict[key]), encoding='utf-8')
    return hash_bytes


def strided(ndarray, stride=1):
    try:
        stride_slice = ndarray.ndim*(slice(None, None, int(stride)),)
    except TypeError:
        assert len(stride) == ndarray.ndim; 'stride must be a tuple of length ndim'
        stride_slice = tuple(slice(None, None, s) for s in stride)
    return ndarray[stride_slice]


def stride_to_size(ndarray, max_size):
    size = array_size(ndarray)
    if size > max_size:
        stride = int(np.ceil((size/max_size)**(1/ndarray.ndim)))
        return strided(ndarray, stride)
    else:
        return ndarray

def press_arr_to_bytes(arr, max_size):
    stride = arr.nbytes//(max_size*1024**2) + 1
    arr_flat = arr.flat
    pressed_len = len(arr_flat)//stride
    new_arr = np.zeros(pressed_len, arr.dtype)
    for k in range(stride):
        new_arr += arr_flat[k::stride][:pressed_len]
    print(f'pressed arr from size {arr.nbytes/1024**2:.2f} to {new_arr.nbytes/1024**2:.2f} MB')
    return new_arr.tobytes()




def array_size(ndarray, units_1024_exp=2):
    # units_1024_exp is 2 fo MB, 3 for GB, ...
    return np.prod(ndarray.shape)*ndarray.dtype.itemsize/1024**units_1024_exp


json_encoder = ExtendedJSONEncoder()


# array save/load functions (for 3D arrays, the format is identical to ImagesReader and ImagesWriter)
def save_ndarray(fname:str, arr:np.ndarray, info:dict=None):
    json_data = dict()
    if info is not None:
        json_data.update(info)
    json_data['shape'] = arr.shape # info entry may be overwritten
    json_data['dtype'] = arr.dtype.str
    with open(replace_suffix(fname, 'info'), 'w') as file:
        json.dump(json_data, file, sort_keys=True, indent=4)
    arr.tofile(fname)


def load_ndarray(fname:str):
    with open(replace_suffix(fname, 'info'), 'r') as file:
        arr_info = json.load(file)
    arr_info.setdefault('info', {})

    return np.fromfile(fname, arr_info['dtype'], count=np.prod(arr_info['shape'])).reshape(arr_info['shape']), arr_info['info']


def separate_suffix(filename):
    split_list = filename.split('.')
    if len(split_list) > 1:
        suffix = split_list[-1]
        return filename[:-(len(suffix)+1)], suffix
    else:
        return filename, ''


def replace_suffix(filename, suffix):
    first_part, old_suffix = separate_suffix(filename)
    if len(old_suffix) > 4: # the thing after the dot is not a file ending
        first_part += "." + old_suffix
    if suffix is None:
        return os.path.normpath(first_part)
    else:
        return os.path.normpath(first_part + '.' + suffix)

