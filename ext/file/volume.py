''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import glob, gc, time, shutil, concurrent.futures as conc_fut, copy

from ext.file.image import *
from ext.image.resize import *
from ext.common.settings_hash import *

#from file.image import *
#from common.settings_hash import *
#from image.resize import *


# images stack access classes
class Reader():
    use_threaded_IO = 0
    modes = 'auto', 'raw', 'raw_stack'

    def __init__(self, path: str, pattern: str='*', mode: str='auto', dtype='f4', return_dtype=None, shape=(0, 0, 0),
                 header: int=0, crops=(None, None, None), value_range=None, do_rescale: bool=False,
                 bin_factor=1, exclude_patterns: list=None, sort_name_range=(None, None), base_folder=None,
                 defer_prepare: bool=False):
        # WARNING: default value for dtype='f4'

        '''
        :param path:                input path, file or folder
        :param pattern:             search pattern for folder
        :param mode:                'auto': binary images (tif, png, ...), 'raw': raw images, 'raw_stack': raw stack
        :param dtype:               write_dtype for raw input
        :param shape:               shape for raw input
        :param header:              header or (header, multiple_header) for raw input, length in bytes
        :param crops:               zyx crop indices for the read image, axis entry can be None (=> (0, 0))
        :param value_range:         value range of uint input (use do_rescale=True)
        :param do_rescale:          rescale input and return as float32
        :param bin_factor:          bin down image on read, must be <= 4
        :param enforce_correct_shape:       enforce correct shape for binary images, prepare can be very time-consuming
        :param raw_enforce_exact_size:      enforce exact size for raw input
        :param progress_signal:             progress signal for ui
        :param restrict_binary_to_IMAGE_FORMATS:    ...
        :param use_IMAGE_IGNORED_TYPES:             ignore files in IMAGE_IGNORED_TYPES
        :return:
        '''
        self.neglog_clip_val = 0.001
        self.allow_data_padding = True
        self.silent = True
        self.ref_treat_nan = True
        self.restrict_binary_to_IMAGE_FORMATS = True
        self.use_IMAGE_IGNORED_TYPES = True
        self.raw_enforce_exact_size = False
        self.enforce_correct_shape = False
        self.progress_signal = None
        self.last_index = 0

        self.batch_load = False
        self.raw_detection_enabled = True
        self.stack_average = None

        self.memmaps_list = []
        self.ref_reader = None
        self.dark_reader = None
        self.ref_image_uncropped = None
        self.dark_image_uncropped = None
        self.first_index = 0
        self.info = None
        self.stop = False
        self.is_raw_return = False

        self.any_transformation = False
        self.plugin_list = []
        self.crops = dict() # crops used internally
        self.crops_external = [(0, 0), (0, 0), (0, 0)] # crops as passed to set_crops() (different if binning enabled)
        self.file_cache_fname = None

        self.keep_volume_result_reference = False
        self.volume = None

        self.ref_dark_hash = ''
        self.ref_dark = None, None

        self.apply_neglog = False
        if self.apply_neglog: self.any_transformation = True

        self.input_images_list = []
        self.images_list = []
        self.input_image = None

        self.ref_reader_hash = None
        self.dark_reader_hash = None
        self.ref_image_ = None
        self.dark_image_ = None

        self.prepared_hash = None
        self.is_prepared = False
        self.ui_settings_hash = None

        self.stack_average = None
        self.stack_average_hash = ''

        self.clear_processing_funcs()

        self.configure(path, pattern, mode, dtype, return_dtype, shape, header, crops, value_range, do_rescale,
                       bin_factor, exclude_patterns, sort_name_range, base_folder, defer_prepare)

        if not defer_prepare:
            self.prepare()

        #print(self, '\n', locals())

    def configure(self, path: str, pattern: str='*', mode: str='auto', dtype='f4', return_dtype=None, shape=(0, 0, 0),
                 header: int=0, crops=(None, None, None), value_range=None, do_rescale: bool=False,
                 bin_factor=1, exclude_patterns: list=None, sort_name_range=(None, None), base_folder=None,
                 defer_prepare: bool=False):
        '''
        :param path:                input path, file or folder
        :param pattern:             search pattern for folder
        :param mode:                'auto': binary images (tif, png, ...), 'raw': raw images, 'raw_stack': raw stack
        :param dtype:               write_dtype for raw input
        :param shape:               shape for raw input
        :param header:              header or (header, multiple_header) for raw input, length in bytes
        :param crops:               zyx crop indices for the read image, axis entry can be None (=> (0, 0))
        :param value_range:         value range of uint input (use do_rescale=True)
        :param do_rescale:          rescale input and return as float32
        :param bin_factor:          bin down image on read, must be <= 4
        :return:
        '''
        if type(path) is not str: print('WARNING: path argument is not a string, wrong arguments?')

        self.reset()
        if type(mode) is int:
            mode = self.modes[mode]
        assert mode in self.modes, f'mode must be one of {self.modes}'
        self.mode = self.modes.index(mode)
        self.path = str(path)
        if pattern == "":
            pattern = "*"
        if '*' not in pattern:
            pattern = pattern + '*'
        self.pattern = pattern

        self.exclude_patterns_init = exclude_patterns if exclude_patterns is not None else []
        self.exclude_patterns_init.append("*_cached_avg_*")
        self.dtype = dtype
        self.return_dtype_ = return_dtype
        self.header = header
        crops = image.cropping.convert_from_old_crop_format(crops)
        self.crops = list(crops)
        self.do_rescale = do_rescale
        self.do_rescale_init = do_rescale
        self.bin_factor = min(bin_factor, 4)
        if do_rescale:
            assert value_range is not None, 'value_range must be given for do_rescale==True'
        self.value_range = value_range
        self.sort_name_range = sort_name_range
        self.base_folder = base_folder

        if len(shape) == 2:
            shape = 0, shape[0], shape[1]
        self.init_shape = shape
        self.input_shape = shape
        self.shape = shape

        if not defer_prepare:
            self.prepare()

        #print(self, '\n', locals())
    # prepare code
    def glob_path(self):
        candidates = self.path, os.path.split(self.path)[0]
        if self.base_folder is not None:
            candidates += pjoin(self.base_folder, self.path), pjoin(self.base_folder, os.path.split(self.path)[0])

        for k, candidate in enumerate(candidates):
            if os.path.isdir(candidate):
                if k % 2 == 0:
                    return glob.glob(pjoin(candidate, self.pattern))
                else:
                    return glob.glob(pjoin(candidate, os.path.split(self.path)[1]+self.pattern))
        else:
            raise FileNotFoundError('images directory does not exist: {}'.format(self.path))

    def prepare(self):
        self.enable_ref_dark_exclude_patterns()
        if self.mode == 0:
            files_list = self.glob_path()
            self.ignore_invalid_filetypes(files_list)
            if self.exclude_patterns is not None:
                do_exclude_patterns(files_list, self.exclude_patterns)
            if len(files_list) == 0:
                error_msg = 'no binary image files found in "{}"'.format(self.path)
                if self.pattern != '*':
                    error_msg += ' with pattern "{}"'.format(self.pattern)
                raise FileNotFoundError(error_msg)
            images_list = []

            nfiles = len(files_list)
            if self.enforce_correct_shape:
                test_indices = np.arange(nfiles)
            else:
                if nfiles > 5:
                    test_indices = np.clip(np.unique(np.array((0, np.random.randint(1, nfiles//2), 
                                                np.random.randint(nfiles//2, nfiles-1), nfiles-1), 'i4')), 0, nfiles-1)
                else:
                    test_indices = np.array((0, nfiles-1), 'i4')


            first_hash = None
            is_raw = False
            if self.enforce_correct_shape:
                for test_index in test_indices:
                    is_valid, imshape, is_raw, im_dtype, nheader = test_binary_image(files_list[test_index])
                    if imshape[0] == 0 or imshape[1] == 0:
                        imshape = load(files_list[test_index]).shape
                    hash = '{} {} {} {}'.format(is_raw, imshape, im_dtype, nheader)
                    if first_hash is None:
                        first_hash = hash
                    else:
                        if first_hash != hash:
                            raise TypeError('images of different type are in the folder {}, pattern {}, {} != {}'.format(
                                              self.path, self.pattern, first_hash, hash))

                    if self.enforce_correct_shape:
                        images_list.append(files_list[test_index])

            else:
                is_valid, imshape, is_raw, im_dtype, nheader = test_binary_image(files_list[0])
                images_list = files_list

            if is_raw and self.raw_detection_enabled:
                self.mode = 1
                self.dtype = im_dtype
                self.header = nheader
                #print('reading tif/edf images as raw with {}'.format(first_hash))

            #is_valid, imshape = self.test_valid_binary_image(files_list[:20][-1])

            self.input_images_list = sort_numbered(images_list, self.sort_name_range)
            if len(self.input_images_list) > 0:
                self.try_load_info()
            self.input_shape = len(self.input_images_list), imshape[0], imshape[1]

        elif self.mode == 1:
            files_list = self.glob_path()
            self.ignore_invalid_filetypes(files_list)
            if self.exclude_patterns is not None:
                do_exclude_patterns(files_list, self.exclude_patterns)
            if len(files_list) == 0:
                error_msg = 'no raw files found in "{}"'.format(self.path)
                if self.pattern != '*':
                    error_msg += ' with pattern "{}"'.format(self.pattern)
                raise FileNotFoundError(error_msg)

            array_nbytes = np.prod(self.input_imshape) * np.dtype(self.dtype).itemsize + self.header
            if self.enforce_correct_shape:
                images_list = []
                for filename in files_list:
                    file_nbytes = os.path.getsize(filename)
                    if self.raw_enforce_exact_size:
                        if file_nbytes == array_nbytes:
                            images_list.append(filename)
                    else:
                        if file_nbytes >= array_nbytes:
                            images_list.append(filename)
            else:
                images_list = files_list

            if len(images_list) == 0:
                error_msg = 'no valid raw images found in "{}"'.format(self.path)
                if self.pattern != '*':
                    error_msg += ' with pattern "{}"'.format(self.pattern)
                raise FileNotFoundError(error_msg)

            self.input_images_list = sort_numbered(images_list, self.sort_name_range)
            if len(self.input_images_list) > 0:
                self.try_load_info()
            self.input_shape = len(self.input_images_list), self.input_imshape[0], self.input_imshape[1]

        elif self.mode == 2:
            candidates = (self.path,)
            if self.base_folder is not None:
                candidates += (pjoin(self.base_folder, self.path),)
            self.ignore_invalid_filetypes(candidates)
            for candidate in candidates:
                if os.path.isfile(candidate):
                    self.path = candidate
                    break
            else:
                raise FileNotFoundError('file not found: {}'.format(self.path))

            self.try_load_info()
            array_nbytes = np.prod(np.int64(self.input_shape)) * np.dtype(self.dtype).itemsize + self.header + self.mheader * self.input_nimages

            self.path_nbytes = os.path.getsize(self.path)
            if self.raw_enforce_exact_size:
                if self.path_nbytes != array_nbytes:
                    raise FileNotFoundError('self.path does not have the rigth size:' + self.path)
            else:
                if self.path_nbytes < array_nbytes:
                    raise OSError('file {} is too small by {} bytes: {}'.format(self.path, array_nbytes-self.path_nbytes, self.path))
            self.input_images_list = (self.path,)

        self.is_prepared = True
        self.set_crops(self.crops)
        #print('reader prepared:\n', self)

    def reset_prepare(self):
        self.is_prepared = False
        if self.ref_reader is not None:
            self.ref_reader.is_prepared = False
        if self.dark_reader is not None:
            self.dark_reader.is_prepared = False

    def reset_crops(self):
        self.set_crops(((0, 0), (0, 0), (0, 0)))

    def set_crops(self, crops=None, bin_factor=None):
        crops = image.cropping.convert_from_old_crop_format(crops)
        if not self.is_prepared: self.prepare()

        if bin_factor is not None:
            self.bin_factor = bin_factor
        for k, crop in enumerate(crops):
            if crop is not None:
                self.crops_external[k] = crop
                self.crops[k] = image.cropping.get_valid_crops_len(crop, self.input_shape[k])

        self.crops = image.cropping.get_positive_crops(self.input_shape, self.crops)
        self.first_index = self.crops[0][0]
        self.shape = image.cropping.get_cropped_shape(self.input_shape, self.crops) # also sets imshape
        if self.mode in (0, 1):
            self.images_list = image.cropping.crop_list(self.input_images_list, self.crops[0])
            assert len(self.images_list) == self.nimages
        else:
            self.images_list = [self.path,]  # behaviour for raw stacks

        if self.mode in (1, 2):
            self.is_raw_return = True
        self.processing_funcs_prepared = False
        assert np.all(self.shape >= 1), "invalid shape for reader:\n{}".format(self)

        correction_crops = [None, self.crops[1], self.crops[2]]
        if self.ref_reader is not None:
            self.ref_reader.set_crops(correction_crops, bin_factor=1)

        if self.dark_reader is not None:
            self.dark_reader.set_crops(correction_crops, bin_factor=1)

    def ignore_invalid_filetypes(self, files_list):
        if self.mode == 0:
            if self.restrict_binary_to_IMAGE_FORMATS:
                for index, filename in list(enumerate(files_list))[::-1]:
                    if not file_in_IMAGE_FORMATS(filename):
                        del files_list[index]
        if self.use_IMAGE_IGNORED_TYPES:
            for index, filename in list(enumerate(files_list))[::-1]:
                if file_in_IMAGE_IGNORED_TYPES(filename):
                    del files_list[index]

    def reset(self):
        self.is_prepared = False
        self.prepared_hash = None
        self.stack_average_hash = None
        if self.ref_reader is not None:
            self.ref_reader.reset()
        if self.dark_reader is not None:
            self.dark_reader.reset()
        self.volume = None

    def hash(self, include_crops=False, include_processing_list=False):
        hash_info = dict(mode=self.mode, path=self.path, pattern=self.pattern, shape=self.input_shape,
                         value_range=self.value_range, do_rescale=self.do_rescale, header=self.header,
                         init_shape=self.init_shape, dtype=self.dtype, bin_factor=self.bin_factor)
        if include_crops:
            hash_info['crops'] = self.crops_external
            hash_info['bin_factor'] = self.bin_factor
        if include_processing_list:
            hash_info['processing_classes'] = [str(plugin.__class__) for plugin in self.plugin_list]
            hash_info['processing_settings'] = self.settings_list
        return generate_hash_from_dict(hash_info)

    def enable_ref_dark_exclude_patterns(self):
        self.exclude_patterns = list(self.exclude_patterns_init) if self.exclude_patterns_init is not None else None
        for reader in (self.ref_reader, self.dark_reader):
            if reader is not None:
                if reader.path == self.path and reader.mode == self.mode and reader.pattern != '*':
                    if self.exclude_patterns is None:
                        self.exclude_patterns = [reader.pattern,]
                    else:
                        try:
                            self.exclude_patterns.append(reader.pattern)
                        except AttributeError:
                            self.exclude_patterns = list(self.exclude_patterns)
                            self.exclude_patterns.append(reader.pattern)

    def do_stop(self):
        self.stop = True

    # properties
    @property
    def header(self):
        try:
            return self._header[0]
        except TypeError:
            return self._header

    @header.setter
    def header(self, val):
        self._header = val

    @property
    def mheader(self):
        try:
            return self._header[1]
        except TypeError:
            return 0

    @property
    def input_shape(self):
        return self._input_volshape

    @input_shape.setter
    def input_shape(self, vals):
        self._input_volshape = np.int64(vals)
        self.input_imshape = np.int64(vals[1:])
        self.input_nimages = np.int64(vals[0])

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, vals):
        if self.bin_factor > 1:
            vals = np.asarray(vals, "i8")
            vals[1:] //= self.bin_factor
        self._shape = np.int64(vals)
        self.nimages = np.int64(vals[0])
        self.imshape = np.int64(vals[1:])

    def emit_progress(self, i , n):
        if self.progress_signal is not None:
            self.progress_signal.emit(int((i+1)/n*100))

    # stack average and caching
    def get_stack_average(self, file_cache=False, block_length=5, stride=1):
        '''
        function for calculating the average or mean of the file stack,
        decides from the number of images when to use a mean or median

        uses median and mean for a large number of images (> 100) (sqrt(n) mean x sqrt(n) median)
        uses a median for a medium number of images (20 - 100)
        uses a mean for a small number of images (< 20)
        :return:
        '''
        if not self.is_prepared:
            self.prepare()
        current_hash = self.hash(False, True)
        if self.stack_average_hash != current_hash:
            stack_average = self.load_stack_average_file(current_hash)

            if stack_average is None:
                crops = copy.deepcopy(self.crops_external)
                bin_factor = self.bin_factor
                self.set_crops([None, (0, 0), (0, 0)], bin_factor=1)
                self.stop = False
                #print(self)
                if self.nimages > 25:
                    nblocks = self.nimages // block_length
                    block_indices = np.arange(0, nblocks, stride)
                    nimages = self.nimages - nblocks # avoid indices above range because of  '+ block_index' below
                    sum_image = np.zeros(self.imshape, 'f8')
                    #print(block_indices, nimages, nblocks)
                    for buffer_index, block_index in enumerate(block_indices):
                        if not self.stop:
                            proj_indices = np.arange(0, nimages, nblocks) + block_index
                            #print(proj_indices)
                            images_block = np.zeros((self.imshape[0], self.imshape[1], len(proj_indices)), dtype='f4')
                            # ! last axis (fastest) are the different images
                            for in_block_index, image_index in enumerate(proj_indices):
                                if not self.stop:
                                    images_block[:, :, in_block_index] = self.load(image_index)
                                if in_block_index % 50 == 0:
                                    gc.collect()
                            sum_image += np.nanmedian(images_block, axis=2, overwrite_input=True)
                            del images_block
                            gc.collect()
                    stack_average = (sum_image / len(block_indices)).astype('f4')
                else:
                    sum_image = np.zeros(self.imshape, 'f8')
                    for k in range(self.nimages):
                        sum_image += self.load(k)
                    stack_average = (sum_image / self.nimages).astype('f4')
                gc.collect()
                if file_cache:
                    self.save_stack_average_file(stack_average, current_hash)
                self.set_crops(crops, bin_factor=bin_factor)

            self.stack_average = stack_average
            self.stack_average_hash = current_hash

        image = crop_arr(self.stack_average, self.crops[1:])
        if self.bin_factor > 1:
            image = bin_down_image(image, self.bin_factor)
        return image

    def load_stack_average_file(self, stack_hash):
        cached_fname = modify_path(self.input_images_list[0], after="_cached_avg_"+stack_hash[:12], suffix="tif")
        if os.path.isfile(cached_fname):
            return load(cached_fname)
        else:
            return None

    def save_stack_average_file(self, stack_average, stack_hash):
        old_files = glob.glob(modify_path(self.input_images_list[0], after="_cached_avg_*", suffix="tif"))
        current_time = time.time()
        for fname in old_files:
            try:
                last_mod_time = os.path.getmtime(fname)
                if current_time - last_mod_time > 60**2*24*2: # older than two days
                    os.remove(fname)
                    print("deleted old stack average:", fname)
            except OSError: pass

        cached_fname = modify_path(self.input_images_list[0], after="_cached_avg_"+stack_hash[:12], suffix="tif")
        try:
            save(stack_average, cached_fname)
            print("saved stack average to", cached_fname, os.getpid())
        except:
            print("saving stack average failed", cached_fname)

    # ref and dark image support
    def set_ref_reader(self, ref_reader=None):
        self.ref_reader = ref_reader
        if ref_reader is not None:
            self.any_transformation = True
            if self.ref_reader_hash != self.ref_reader.hash():
                self.ref_image_ = None
                self.ref_reader_hash = None
                self.ref_image_uncropped = None
            self.check_rescale()

    def set_dark_reader(self, dark_reader=None):
        self.dark_reader = dark_reader
        if dark_reader is not None:
            self.any_transformation = True
            if self.dark_reader_hash != self.dark_reader.hash():
                self.dark_image_ = None
                self.dark_reader_hash = None
                self.dark_image_uncropped = None
            self.check_rescale()

    def check_rescale(self):
        if self.dark_reader is not None or self.ref_reader is not None:
            self.do_rescale = False

    def ref_image(self):
        if self.ref_reader is not None:
            ref = self.ref_reader.get_stack_average(file_cache=True)
            dark = self.dark_image()
            if dark is not None:
                ref -= dark
            if self.ref_treat_nan:
                np.copyto(ref, ref.mean(), where=np.isclose(ref, 0))
            return ref
        else:
            return None

    def dark_image(self):
        if self.dark_reader is not None:
            return self.dark_reader.get_stack_average(file_cache=True)
        else:
            return None

    def ref_dark_images(self):
        if self.ref_reader is not None:
            ref_dark_hash = self.ref_reader.hash(True, True)
        else:
            ref_dark_hash = 'none'
        if self.dark_reader is not None:
            ref_dark_hash += self.dark_reader.hash(True, True)
        else:
            ref_dark_hash += 'none'

        if ref_dark_hash != self.ref_dark_hash:
            self.ref_dark_hash = ref_dark_hash

            if self.ref_reader is not None:
                ref = np.copy(self.ref_reader.get_stack_average(file_cache=True))
            else:
                ref = None
            dark = np.copy(self.dark_image())
            if dark is not None:
                ref -= dark
            if ref is not None and self.ref_treat_nan:
                np.copyto(ref, ref.mean(), where=np.isclose(ref, 0))  # quick and dirty
            self.ref_dark = ref, dark
            return ref, dark
        else:
            return self.ref_dark[0], self.ref_dark[1]

    # raw access functions
    def generate_raw_cache(self, filename=None, open_as_memmap=False):
        if not self.is_raw_return:
            if not self.is_prepared:
                self.prepare()
            if self.mode == 0:
                self.dtype = self.load(0).dtype
            if filename is None:
                self.file_cache_fname = modify_path(self.path, after='_read_cache', suffix='raw')
            else:
                self.file_cache_fname = filename

            writer = Writer(2, self.file_cache_fname, write_dtype=self.return_dtype, shape=self.shape)
            for k in range(self.nimages):
                writer.save(self.load(k), k)

    def delete_raw_cache(self):
        if self.file_cache_fname is not None:
            val = try_delete_file(self.file_cache_fname)
            if val:
                self.file_cache_fname = None
            return val
        else:
            return True

    def get_raw_reader(self):
        # to generate a reader that can then be inherited by a subprocess and which will have fast access to subareas
        if not self.is_prepared:
            self.prepare()
        if self.is_raw_return:  # if a reader can be a memmap, it can also efficiently load subareas of the volume
            return self
        else:
            self.generate_raw_cache(open_as_memmap=False)
            return Reader(self.file_cache_fname, shape=self.shape, dtype=self.return_dtype, do_rescale=self.do_rescale,
                          value_range=self.value_range, bin_factor=self.bin_factor, mode='raw_stack')

    def get_settings(self):
        reader_settings = dict()
        reader_settings['mode'] = self.mode
        reader_settings['path'] = self.path
        reader_settings['pattern'] = self.pattern
        reader_settings['dtype'] = self.dtype
        reader_settings['do_rescale'] = self.do_rescale
        reader_settings['value_range'] = self.value_range
        reader_settings['bin_factor'] = self.bin_factor
        reader_settings['value_range'] = self.value_range
        reader_settings['shape'] = self.shape
        reader_settings['headers'] = self.header, self.mheader
        reader_settings['crops'] = self.crops
        return reader_settings

    @property
    def return_dtype(self):
        if self.any_transformation or self.do_rescale:
            return 'f4'
        else:
            return self.return_dtype_

    @return_dtype.setter
    def return_dtype(self, dtype_):
        self.return_dtype_ = np.dtype(dtype_).str

    # image stack read functions
    def load_all(self):
        if not self.is_prepared:
            self.prepare()
        return self.load_range((0, self.nimages))

    def load_range(self, indices):
        self.batch_load = True
        self.stop = False
        if not self.is_prepared:
            self.prepare()
        assert indices[1] > indices[0]
        volume = np.zeros((indices[1]-indices[0], self.shape[1], self.shape[2]), dtype=self.return_dtype)
        collect_number = int(self.nimages/(volume.nbytes/1024**3))
        # print('self.vol_shape', self.vol_shape, volume.shape, self.crops)
        if self.use_threaded_IO:
            with conc_fut.ThreadPoolExecutor(self.use_threaded_IO) as tpool:
                for k_0, k_z in enumerate(np.arange(*indices)):
                    tpool.submit(self.load_to_volume, volume, k_z, k_0)

        else:
            for k_0, k_z in enumerate(np.arange(*indices)):
                if self.stop: return volume
                self.load_to_volume(volume, k_z, k_0)
                #self.emit_progress(index, self.nimages)
                if (k_0+1) % collect_number == 0:
                    gc.collect()

        if collect_number < 1000:
            gc.collect()

        self.batch_load = False

        if self.keep_volume_result_reference:
            self.volume = volume
        else:
            self.volume = None
        return volume

    def load_to_volume(self, volume, k_z, k_0):
        volume[k_0, :, :] = self.load(k_z)

    def load_indices(self, indices_list):
        self.batch_load = True
        if not self.is_prepared:
            self.prepare()
        volume = np.zeros((len(indices_list), self.imshape[0], self.imshape[1]), dtype=self.return_dtype)
        collect_number = int(self.nimages/(volume.nbytes/1024**3))
        # print('self.vol_shape', self.vol_shape, volume.shape, self.crops)
        if self.use_threaded_IO:
            print('using threaded read', self.use_threaded_IO)
            self.volume = volume
            with conc_fut.ThreadPoolExecutor(self.use_threaded_IO) as tpool:
                for k_0, k_z in enumerate(indices_list):
                    tpool.submit(self.load_to_volume, k_z, k_0)
            volume = self.volume
            self.volume = None

        else:
            for k_0, k_z in enumerate(indices_list):
                volume[k_0, :, :] = self.load(k_z)
                #self.emit_progress(index, self.nimages)
                if (k_0+1) % collect_number == 0:
                    gc.collect()
        if collect_number < 1000:
            gc.collect()
        self.batch_load = False
        return volume


    # info file treatment
    def try_load_info(self, use_settings=True):
        #print("try_load_info")
        if self.mode == 2:
            info = load_info(self.path)
            #print("info_filename", info_filename, info)

        else:
            nimages = len(self.input_images_list)-1
            info = load_info([self.input_images_list[k] for k in (0, nimages//3, 2*nimages//3, nimages)], optional=True)
            if info is None:
                possible_info_filenames = glob.glob(pjoin(self.path, "*.info"))
                path_overlaps = np.asarray([len(os.path.commonprefix((info_filename, self.input_images_list[0]))) for info_filename in possible_info_filenames])

                for k in np.argsort(path_overlaps)[::-1]:
                    info = load_info(possible_info_filenames[k])
                    if info is not None:
                        break

        if info is not None and use_settings:
            try:
                if type(info['shape']) is str:
                    info['shape'] = np.asarray(list(info['shape']), "i8")
                input_shape = np.zeros(3, dtype='i8')
                input_shape[1:] = info['shape'][:2][::-1]
                if len(info['shape']) == 3:
                    input_shape[0] = info['shape'][2]
                self.input_shape = input_shape
                self.dtype = info['dtype']
            except (ValueError, KeyError):
                self.has_info = False
                return
            try:
                self.value_range = info['value_range']
                self.do_rescale = True
            except KeyError:
                pass
            #print("reader is using info", info)

        self.has_info = True
        self.info = info
        # must call set_crops() to be effective
        return info


    # functions for applying functions to individual 2D-images (applied in load())
    def set_processing_funcs(self, plugin_list, settings_list, ctx_info=None, batch=True):
        assert len(plugin_list) == len(settings_list), 'plugin_list and settings_list must be of equal length'
        self.plugin_list = plugin_list
        self.settings_list = settings_list
        self.ctx_info = ctx_info
        self.processing_funcs_prepared = False
        self.batch = batch
        self.processing_filter_range = 0.
        if len(plugin_list) > 0:
            self.any_transformation = True

    def prepare_processing_funcs(self):
        # print('self.settings_list', self.settings_list, len(self.settings_list))
        filters_list = []
        is_fourier = False
        try:
            image_shape = self.imshape
        except AttributeError:
            image_shape = self.reader.imshape # for ProjectionsReadThread
        for plugin_index in range(len(self.settings_list)):
            if self.settings_list[plugin_index]['fourier_inout'][0]:  # means plugin_list[plugin_index] is a get_filter function
                if is_fourier:
                    filter_calls.append((self.plugin_list[plugin_index].get_filter, self.settings_list[plugin_index]))
                    filterer.add_pad_length(self.settings_list[plugin_index]['filter_range'])
                    self.processing_filter_range += self.settings_list[plugin_index]['filter_range']
                else:
                    #print('used numpy fallback for small fft', False if prod(image_shape) > 1e6 else True)
                    from ext.image.fourier import FourierFilterer
                    filterer = FourierFilterer(batch=self.batch)
                    filter_calls = [(self.plugin_list[plugin_index].get_filter, self.settings_list[plugin_index]),]
                    filterer.pad_length = self.settings_list[plugin_index]['filter_range']
                    self.processing_filter_range += self.settings_list[plugin_index]['filter_range']

                    is_fourier = True
            if not self.settings_list[plugin_index]['fourier_inout'][1]:
                self.settings_list[plugin_index]["parallelism"] = os.cpu_count()
                if is_fourier:
                    filterer.set_filter(filter)
                    filters_list.append((filterer, True, filter_calls, image_shape))
                    is_fourier = False
                filters_list.append((plugin_index, False))
                if self.plugin_list[plugin_index].changes_shape:
                    image_shape = self.plugin_list[plugin_index].process_image(np.zeros(image_shape, dtype='f4'), 0,
                                                                               self.settings_list[plugin_index]).shape

        if is_fourier:
            filters_list.append((filterer, True, filter_calls, image_shape))

        # fourier filter setup must be after all range calculations because padding changes the image shape
        for index, entry in enumerate(filters_list):
            if entry[1]:
                filterer, is_fourier, filter_calls, image_shape = entry
                filterer.prepare_shape(image_shape)
                filter = 1.
                for (filter_call, filter_settings) in filter_calls:
                    filter = filter*filter_call(filterer.padded_shape, filterer.u_axes(), filter_settings)
                filterer.set_filter(filter)
                filters_list[index] = (filterer, True)

        if self.allow_data_padding and self.processing_filter_range > 0.:
            self.data_padding_ranges = np.zeros((2, 2), 'u2')
            for k in range(2):
                if self.crops[k+1][0] > 0:
                    self.data_padding_ranges[k, 0] = min(self.crops[k+1][0], self.processing_filter_range)
                if self.crops[k+1][1] < self.input_imshape[k]:
                    self.data_padding_ranges[k, 0] = min(self.input_imshape[k]-self.crops[k+1][0], self.processing_filter_range)
            self.padding_ranges = int(np.ceil(self.processing_filter_range)) - self.data_padding_ranges
            #print('self.data_padding_ranges, self.padding_ranges', self.data_padding_ranges, self.padding_ranges)
            self.use_data_padding = True
        else:
            self.use_data_padding = False

        #print('filters_list', filters_list)
        self.filters_list = filters_list
        self.processing_funcs_prepared = True

    def clear_processing_funcs(self):
        self.plugin_list = []
        self.settings_list = []

    def apply_transformations(self, image, image_index):
        self.input_image = image
        if len(self.plugin_list) > 0:
            t0 = time.time()
            if not self.processing_funcs_prepared:
                self.prepare_processing_funcs()

            for filter, is_fouriermult in self.filters_list: # filter is either a index for plugin_list or a FourierFilterer
                if is_fouriermult:
                    image = filter.apply_filter(image)
                    #print('fftnmult applied')
                else:
                    image = self.plugin_list[filter].process_image(image, image_index, self.settings_list[filter])
                    #print('plugin filter applied')
            #print('apply_plugin_list runtime: {:.2f} s'.format(time.time()-t0))
            gc.collect()
        return image

    def correct_ref_dark(self, image):
        image = image.astype("f4", copy=False)
        ref_image, dark_image = self.ref_dark_images()
        if dark_image is not None:
            image -= dark_image
        if ref_image is not None:
            image /= ref_image

        if self.apply_neglog:
            np.fmax(image, self.neglog_clip_val, out=image)
            np.log(image, out=image)
            image *= -1

        return image

    def delete_files(self):
        successes = []
        for file in self.input_images_list:
            successes.append(try_delete_file(file))
        return np.all(successes)

    # base load function
    def load(self, index):
        # WARNING: index is a local index, 0 is the first of the cropped stack
        try:
            del self.input_image
        except AttributeError: pass
        if not self.is_prepared:
            self.prepare()
        if index is "center":
            index = self.nimages // 2
        index = np.clip(index, 0, self.input_nimages-1)

        if self.mode == 0:
            with open(self.images_list[index], mode='rb') as file:
                pil_image = PIL.Image.open(file)
                pil_image = crop_pil_image(pil_image, self.crops, self.imshape)
                image = np.array(pil_image)
                #image = copy(crop_2D(image, self.crops['ycrop'], self.crops['xcrop']))

            if image.ndim == 3:
                image = np.mean(image, axis=2)  # gray values are calculated from the average over the last axis

        elif self.mode == 1:
            with open(self.images_list[index], mode='rb') as file:
                dtype_nbytes = np.dtype(self.dtype).itemsize
                try:
                    file.seek(self.header + dtype_nbytes*self.crops[1][0]*self.input_imshape[1])
                except OSError:
                    raise RuntimeWarning('seek error ignored')

                read_shape = get_cropped_shape(self.input_imshape, [self.crops[1], (0, 0)])
                image = np.fromfile(file, dtype=self.dtype, count=(read_shape[0]*read_shape[1]))
                if np.prod(read_shape) != image.shape[0]:
                    too_short_bytes = np.prod(self.input_imshape)*np.dtype(self.dtype).itemsize + self.header - os.path.getsize(self.images_list[index])
                    raise ValueError('raw file at {} is too short by {} bytes)'.format(self.images_list[index], too_short_bytes))
                image = image.reshape(read_shape)
                image = np.copy(crop_arr(image, [None, self.crops[2]]))

        elif self.mode == 2:
            dtype_nbytes = np.dtype(self.dtype).itemsize
            index = index + self.first_index
            s = np.int64(0)
            s += np.int64(self.header + dtype_nbytes*self.crops[1][0]*self.input_imshape[1])
            s += np.prod(self.input_imshape)*dtype_nbytes*index + self.mheader*(index+1)
            read_shape = get_cropped_shape(self.input_imshape, [self.crops[1], None])

            with open(self.path, mode='rb') as file:
                try:
                    file.seek(s)
                except OSError:
                    raise RuntimeWarning('seek error ignored')

                image = np.fromfile(file, dtype=self.dtype, count=np.prod(read_shape))
                if np.prod(read_shape) != image.shape[0]:
                    too_short_bytes = np.prod(self.input_shape)*np.dtype(self.dtype).itemsize + self.header + self.nimages*self.mheader - os.path.getsize(self.path)
                    raise ValueError('raw file at {} is too short by {} bytes)'.format(self.path, too_short_bytes))
                image = image.reshape(read_shape)

            image = np.copy(crop_arr(image, [None, self.crops[2]]))

        if self.do_rescale:
            image = unscale_from_dtype(image, self.value_range, image.dtype)
        else:
            image = image.astype(self.return_dtype, copy=False)

        if self.any_transformation:
            image = self.correct_ref_dark(image)
            image = self.apply_transformations(image, index)

        if self.bin_factor > 1:
            image = bin_down_image(image, self.bin_factor)

        self.last_index = index
        return image

    def __str__(self):
        if self.mode == 0:
            mode_text = 'binary images'
        elif self.mode == 1:
            mode_text = 'raw images'
        elif self.mode == 2:
            mode_text = 'raw stack'
        else:
            mode_text = 'invalid mode'

        if self.mode == 2:
            path = self.path
        else:
            path = pjoin(self.path, self.pattern)
        if self.do_rescale:
            rescale_text = ', scale from: {}'.format(self.value_range)
        else:
            rescale_text = ''

        return 'Reader, {} in: {}\nshape: {} => {}, dtype: {}{}, crops: {}'.format(
            mode_text, path, self.input_shape, self.shape, self.dtype, rescale_text, self.crops)


class Writer():
    use_threaded_IO = 0
    modes = 'tif', 'raw', 'raw_stack', 'png'

    def __init__(self, path: str, mode: str='tif', do_rescale=False, value_range=None, write_dtype='f4',
                 number_format='{:04}', filenames=None, base_filename=None, after='', before='',
                 makedir=True, cleardir=False, bin_factor=1, resolution=None, shape=None):

        self.png_compress_level = 2
        self.nonscaling_dtypes = (np.uint8, np.uint16)

        self.stack_lock = None
        self.progress_signal = None

        self.opened = False
        self.truncate_stack = True
        self.info_saved = False
        self.stop = False
        self.file_cache_fname = None

        self.memmap_open = False
        self.files = []
        self.file = None
        self.header = None
        self.nheader = 0

        self.thread_pool = None

        self.any_transformation = False
        if type(mode) is int:
            if mode == 1: mode = 'raw'
            elif mode == 2: mode = 'raw_stack'
            elif mode == 0: mode = 'tif'
        assert mode in self.modes, f'mode must be one of {self.modes}'
        self.mode = mode
        self.suffix = mode
        self.path = path
        self.after = after
        self.before = before
        self.write_dtype = np.dtype(write_dtype).str
        self.do_rescale = do_rescale
        self.value_range = value_range
        self.bin_factor = bin_factor
        self.resolution = resolution
        self.shape = shape

        if do_rescale:
            assert value_range is not None, 'rescale_range must be given for do_rescale == True'
        self.value_range = value_range
        if self.progress_signal is not None:
            self.emit_progress = self.progress_signal.emit
        else:
            self.emit_progress = None
        self.filenames = None
        if filenames is not None:
            self.filenames = [pjoin(path, modify_path(os.path.basename(filename), self.suffix, before, after=after))
                              for filename in filenames]
            self.format_str = None
        elif base_filename is not None:
            self.format_str = modify_path(os.path.basename(base_filename), self.suffix, before, after=after+'_'+number_format)
        elif before=='' and after=='':
            self.format_str = modify_path('', self.suffix, '', after=number_format)
        else:
            self.format_str = modify_path('', self.suffix, before, after=after+'_'+number_format)

        self.makedir = makedir
        self.cleardir = cleardir

        self.header = None
        self.nheader = np.int64(0)

        self.info = None
        self.other_info = dict()
        self.has_info = False
        if self.mode in ('png', "jpg", "jpeg", "gif"):
            self.is_raw_write = False
        else:
            self.is_raw_write = True

        if mode != 'raw_stack':
            if cleardir:
                try:                        shutil.rmtree(path)
                except FileNotFoundError:   pass
            if makedir:
                assure_dir_exists(path)

        else:
            self.imshape = None
            self.nimages = 0
            #self.open()
            #self.can_be_memmap = True
        #print('writer initilized:\n', self)
        self.opened = False
        self.plugin_list = []

    def set_stack_header(self, header, file_ending):
        self.header = bytes(header)
        if header is not None:
            self.nheader = np.int64(len(self.header))
            self.path = replace_suffix(self.path, file_ending)
        else:
            self.nheader = np.int64(0)

        self.open()
        self.file.seek(0)
        self.file.write(self.header)

    # raw access code
    def transfer_raw_cache(self, delete=True):
        if self.file_cache_fname is not None:
            reader = Reader(self.file_cache_fname, dtype=self.write_dtype, shape=self.shape,
                            mode='raw_stack', defer_prepare=False)
            for k in range(reader.nimages):
                self.save(reader.load(k), k)
            if delete:
                val =  try_delete_file(self.file_cache_fname)
                if val:
                    self.file_cache_fname = None
                return val
            else:
                return True

    def get_raw_writer(self, shape=None):
        # to generate a writer that can then be inherited by a subprocess, will have fast write on subareas
        if shape is not None:
            self.shape = shape
        assert self.shape is not None
        if self.is_raw_write:
            if self.mode == 'tif':
                self.nheader, self.header = get_tif_header(self.shape[1:], self.write_dtype)
                self.mode = 'raw'
            self.open()
            self.close()
            return self
        else:
            self.file_cache_fname = modify_path(self.path, after='_write_cache', suffix='raw')
            return Writer(self.file_cache_fname, write_dtype=self.write_dtype, mode='raw_stack',
                          do_rescale=self.do_rescale, value_range=self.value_range, shape=self.shape)

    def write_headers(self):
        if self.header is not None:
            if len(self.files) > 0:
                for file in self.files:
                    file.seek(0)
                    file.write(self.header)
            else:
                self.file.seek(0)
                self.file.write(self.header)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, vals):
        if vals is not None:
            if self.bin_factor > 1:
                vals = np.asarray(vals, "i8")
                vals[1:] //= self.bin_factor
            self._shape = np.int64(vals)
            self.nimages = np.int64(vals[0])
            self.imshape = np.int64(vals[1:])
        else:
            self._shape = None
            self.nimages = 0
            self.imshape = None

    def save_all(self, volume, index_offset=0, other_info=None):
        t0 = time.time()
        self.save_range(volume, (0, volume.shape[0]), index_offset, other_info)
    save_all_par = save_all

    def save_range(self, volume, vol_slice, index_offset=0, other_info=None):
        if self.use_threaded_IO > 0:
            with conc_fut.ThreadPoolExecutor(self.use_threaded_IO) as thread_pool:
                image = self.do_save_range(volume, vol_slice, index_offset, other_info, thread_pool=thread_pool)
        else:
            image = self.do_save_range(volume, vol_slice, index_offset, other_info)
        self.save_info(image)

    def do_save_range(self, volume, vol_slice, index_offset=0, other_info=None, thread_pool=None):
        self.stop = False
        if self.mode == 'raw_stack':
            self.open()
        if self.bin_factor > 1:
            for loc_index, glob_index in list(enumerate(np.arange(*vol_slice)))[::self.bin_factor]:
                image = volume[loc_index:loc_index+self.bin_factor, :, :].mean(axis=0)
                image = bin_down_image(image, self.bin_factor)
                if self.mode == 'raw_stack':
                    write_index = glob_index//self.bin_factor
                    index_offset = index_offset//self.bin_factor
                else:
                    write_index = glob_index+(self.bin_factor-1)//2
                if self.stop: return
                if thread_pool is not None:
                    thread_pool.submit(self.save, image, write_index, index_offset, other_info)
                else:
                    self.save(image, write_index, index_offset, other_info=other_info)
                if (loc_index+1) % 100 == 0: gc.collect()
            self.save_info(image)
        else:
            for loc_index, glob_index in enumerate(np.arange(*vol_slice)):
                if self.stop: return
                if thread_pool is not None:
                    thread_pool.submit(self.save, volume[loc_index, :, :], glob_index, index_offset, other_info)
                else:
                    self.save(volume[loc_index, :, :], glob_index, index_offset, other_info=other_info)
                    if (loc_index+1) % 100 == 0: gc.collect()
            image = volume[0]
            #self.save_info(volume[0])
        gc.collect()
        return image

    def get_filename(self, index):
        if self.mode == 'raw_stack':
            return self.path
        else:
            if self.format_str is not None:
                filename = pjoin(self.path, self.format_str.format(index))
            else:
                filename = self.filenames[index]
            return filename

    def save_info(self, image=None, other_info=None, fail_silent=True):
        #if index == 0: print('save_info()',image.shape, image.write_dtype, other_info)
        if image is None:
            if self.imshape is None:
                if fail_silent:
                    return
                else:
                    raise AssertionError('image=None only allowed if an image was written before or shape is defined')
            imshape = self.imshape
        else:
            imshape = image.shape

        if self.nimages is not None:
            shape = self.nimages, *imshape
        else:
            shape = imshape

        if self.do_rescale:
            self.other_info['value_range'] = self.value_range

        if other_info is not None:
            self.other_info.update(other_info)

        info = (str(shape) + str(self.write_dtype)).encode() + (generate_hash_from_dict(other_info).encode() if other_info is not None else b'')

        if info != self.info:
            if self.mode == 'raw_stack':
                filename = self.path
            else:
                if self.format_str is not None:
                    filename = [pjoin(self.path, self.format_str.format(k)) for k in (0, 999999)]
                else:
                    filename = [self.filenames[k] for k in (0, 999999)]

            save_info(filename, shape, self.write_dtype, **self.other_info)
            self.info = info

    def enable_keep_scaling(self, reader):
        if reader.do_rescale:
            self.value_range = reader.value_range
            self.write_dtype = reader.dtype
            self.do_rescale = True
        else:
            self.do_rescale = False

    def open(self, shape=None, clear_files=False, write_headers=True):
        if not self.opened:
            if shape is not None:
                self.shape = shape

            if clear_files:
                self.clear_files()

            if self.mode == 'tif' and self.shape is not None:
                self.nheader, self.header = get_tif_header(self.shape[1:], self.write_dtype)
                self.mode = 'raw'

            if self.mode == 'raw_stack':
                del self.files[:]
                if not self.truncate_stack:
                    self.file = open(self.path, mode='r+b')
                else:
                    self.file = open(self.path, mode='w+b')
            elif self.mode == 'raw' and self.shape is not None:
                self.file = None
                for index in np.arange(self.shape[0]):
                    fname = self.get_filename(index)
                    try:
                        self.files.append(open(fname, mode='r+b'))
                    except FileNotFoundError:
                        self.files.append(open(fname, mode='w+b'))

            if write_headers:
                self.write_headers()

            gc.collect()
            self.opened = True

    def clear_files(self):
        if self.mode == 'raw_stack':
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
        else:
            for index in np.arange(self.shape[0]):
                try:
                    os.remove(self.get_filename(index))
                except FileNotFoundError:
                    pass

    def close(self):
        if self.imshape is not None:
            self.save_info(None)
        if self.opened:
            if self.mode == 'raw_stack':
                self.file.close()
                self.file = None
            elif self.shape is not None:
                [file.close() for file in self.files]
                del self.files[:]

            self.opened = False

    def check_if_overwriting(self):
        if self.mode == 'raw_stack':
            return os.path.exists(self.path)
        else:
            suffix = '*.' + self.suffix
            files_list = glob.glob(pjoin(self.path, suffix))
            return len(files_list) > 0

    # functions for applying functions to individual 2D-images (applied in save())
    def set_processing_funcs(self, plugin_list, settings_list, ctx_info=None, batch=True):
        self.any_transformation = True
        assert len(plugin_list) == len(settings_list), 'plugin_list and settings_list must be of equal length'
        self.plugin_list = plugin_list
        self.settings_list = settings_list
        self.ctx_info = ctx_info
        self.batch = batch

    def clear_processing_funcs(self):
        self.any_transformation = False
        self.plugin_list = []
        self.settings_list = []

    def apply_transformations(self, image, image_index):
        if len(self.plugin_list) > 0:
            for index, plugin in enumerate(self.plugin_list): # filter is either a index for plugin_list or a FourierFilterer
                image = plugin.process_image(image, image_index, self.settings_list[index])
            gc.collect()

        return image

    # base save function
    def save(self, image, index, index_offset=0, other_info=None):
        if image is None: return

        image = self.apply_transformations(image, index)

        if image.dtype in self.nonscaling_dtypes:
            if self.write_dtype != image.dtype:
                self.write_dtype = image.dtype
        elif self.do_rescale:
            image = scale_to_dtype(image, self.value_range, self.write_dtype)
        else:
            image = image.astype(self.write_dtype, copy=False)

        if self.imshape is None:
            self.imshape = image.shape
        else:
            assert np.all(np.array(self.imshape) == np.array(image.shape)), 'all image shapes must be identical'

        if not self.info_saved:
            self.save_info(image, other_info=other_info)
            self.info_saved = True

        index = index - index_offset
        self.nimages = max(index+1, self.nimages)
        if self.mode == "tif":
            if self.header is None:
                self.nheader, self.header = get_tif_header(image.shape, self.write_dtype)
                self.mode = 'raw'
                #print("switched to raw writer for tif images")

            filename = self.get_filename(index+index_offset)
            pil_args = dict()
            if filename[-3:] == 'png':
                pil_args['compress_level'] = self.png_compress_level
            elif filename[-3:] == 'tif' and self.resolution is not None:
                pil_args['resolution'] = 1/self.resolution*1e4
                pil_args['x_resolution'] = 1/self.resolution*1e4
                pil_args['y_resolution'] = 1/self.resolution*1e4
                pil_args['resolution_unit'] = "cm"
            PIL.Image.fromarray(image).save(filename, **pil_args)

        elif self.mode == 'raw':
            if len(self.files) > 0:
                self.files[index].seek(0)
                if self.header is not None:
                    self.files[index].write(self.header)
                self.files[index].write(image.tobytes())
            else:
                with open(self.get_filename(index), 'w+b') as fp:
                    if self.header is not None:
                        fp.write(self.header)
                    fp.write(image.tobytes())

        elif self.mode == 'raw_stack':
            self.open()
            if self.stack_lock is not None: self.stack_lock.acquire()
            self.file.seek(self.nheader + np.int64(image.shape[0]) * image.shape[1] * index * np.dtype(self.write_dtype).itemsize, 0)
            self.file.write(image.tobytes())
            if self.stack_lock is not None: self.stack_lock.release()

    def write_vol_zy_range(self, volume, z_range, y_range):
        if self.stack_lock is not None: self.stack_lock.acquire()
        for local_index, global_index in enumerate(np.arange(*z_range)):
            self.write_im_y_range(volume[local_index, :, :], global_index, y_range)
        if self.stack_lock is not None: self.stack_lock.release()

    def write_im_y_range(self, image, index, y_range):

        if image.dtype in self.nonscaling_dtypes:
            if self.write_dtype != image.dtype:
                self.write_dtype = image.dtype
        elif self.do_rescale:
            image = scale_to_dtype(image, self.value_range, self.write_dtype)
        else:
            image = image.astype(self.write_dtype, copy=False)

        if self.mode == 'raw':
            self.files[index].seek(self.nheader + self.shape[2]*y_range[0]*np.dtype(self.write_dtype).itemsize, 0)
            self.files[index].write(image.tobytes())
            #self.files[index].flush()

        elif self.mode == 'raw_stack':
            self.file.seek(self.nheader + (np.int64(self.shape[1])*self.shape[2]*index+self.shape[2]*y_range[0])*np.dtype(self.write_dtype).itemsize, 0)
            self.file.write(image.tobytes())
            #self.file.flush()

        else:
            raise ValueError("self.mode must be in ('tif', 'raw'), is: {}".format(self.mode))

    def __str__(self):
        if self.mode == 'raw_stack':
            path = self.path
        elif self.filenames is not None:
            path = self.filenames[0]
        else:
            path = pjoin(self.path, self.format_str)
        if self.do_rescale:
            rescale_text = 'scale to: {}'.format(self.value_range)
        else:
            rescale_text = ''

        return 'Writer in mode {} to path: {},\n dtype: {}, shape: ({}, {}, {}), {}'.format(
            self.mode, path, self.write_dtype, self.nimages, *self.imshape, rescale_text)


class ReadSettings(dict):
    # todo: implement
    pass


class WriteSettings(dict):
    # todo: implement
    pass


def get_reader(input_settings, old_reader=None, progress_signal=None):
    if 'crops' in input_settings:
        input_settings['crops'] = convert_from_old_crop_format(input_settings['crops'])
    if old_reader is not None:
        old_reader.set_ref_reader(None)
        old_reader.set_dark_reader(None)
    try:
        reader = get_base_reader(input_settings['projs'], old_reader, progress_signal)
        if input_settings['use_refs']:
            ref_reader = get_base_reader(input_settings['refs'], reader.ref_reader)
            reader.set_ref_reader(ref_reader)
        else:
            reader.set_ref_reader(None)
        if input_settings['use_darks']:
            dark_reader = get_base_reader(input_settings['darks'], reader.dark_reader)
            reader.set_dark_reader(dark_reader)
        else:
            reader.set_dark_reader(None)
    except KeyError:
        reader = get_base_reader(input_settings, old_reader, progress_signal)
        reader.set_ref_reader(None)
        reader.set_dark_reader(None)
    return reader


def get_base_reader(settings, old_reader:Reader=None, progress_signal=None):
    reader_settings = partial_dict_copy(settings, ('mode', 'path', 'pattern', 'dtype', 'do_rescale', 'value_range',
                                                   'bin_factor', 'crops'))
    reader_settings['shape'] = settings['nimages'], *settings['shape']
    reader_settings['header'] = settings['headers']

    if old_reader is not None:
        if old_reader.ui_settings_hash != generate_hash_from_dict(reader_settings):
            old_reader.configure(**reader_settings, defer_prepare=True)
            old_reader.progress_signal = progress_signal
        else:
            old_reader.set_crops(settings['crops'])
        reader = old_reader
    else:
        reader = Reader(**reader_settings, defer_prepare=True)
        reader.progress_signal = progress_signal

    reader.ui_settings_hash = generate_hash_from_dict(reader_settings)
    return reader


def update_reader(settings, old_reader, progress_signal=None):
    assert old_reader is not None
    get_reader(settings, old_reader, progress_signal)


def get_writer(settings, base_filename=None, makedir=True):
    kwargs = dict(path=settings['path'], base_filename=base_filename,
                  write_dtype=settings['dtype'], mode=settings['mode'], makedir=makedir)
    try:
        kwargs['do_rescale'] = settings['do_rescale']
        kwargs['value_range'] = settings['value_range']
        #print('rescaling set to', kwargs['value_range'], settings['write_dtype'])
    except KeyError:
        kwargs['do_rescale'] = False
    if settings.get('save_png', False) and kwargs['mode'] == 0:
        kwargs['format'] = 'png'

    return Writer(**kwargs)


# stand-ins for the writer/reader classes with arrays instead of files, not all functions are supported
class DummyReader():
    mode = -1

    def __init__(self, arr):
        self.arr = arr
        self.reader = self
        self.input_shape = arr.shape
        self.input_imshape = arr.shape
        self.files_indices = np.arange(arr.shape[0])
        self.set_crops([(0, 0), (0, 0), (0, 0)])

    @property
    def input_shape(self):
        return self._input_volshape

    @input_shape.setter
    def input_shape(self, vals):
        self._input_volshape = np.int64(vals)
        self.input_imshape = np.int64(vals[1:])
        self.input_nimages = np.int64(vals[0])

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, vals):
        self._shape = np.int64(vals)
        self.nimages = np.int64(vals[0])
        self.imshape = np.int64(vals[1:])


    def set_crops(self, crops):
        self.crops = get_valid_crops(crops, self.arr.shape)
        self.shape = get_cropped_shape(self.input_shape, crops)
        self.imshape = self.shape[1:]
        self.shape = get_cropped_shape(self.input_shape, self.crops) # also sets imshape

    def reset_crops(self):
        self.set_crops([(0, 0), (0, 0), (0, 0)])

    def reset(self):
        pass

    def load(self, index):
        return crop_arr(self.arr[self.crops[0][0]+index], [self.crops[1], self.crops[2]])

    def load_all(self):
        self.volume = crop_arr(self.arr, self.crops)
        return self.volume


class DummyWriter():
    mode = -1

    def __init__(self, arr):
        self.arr = arr
        self.writer = self
        self.nwrites = 0
        self.mode = 2

    def save(self, image, slice_number, index_offset=0):
        if self.arr is not None:
            self.arr[slice_number+index_offset, :, :] = image
        self.nwrites += 1

    def save_range(self, volume, vol_slice, index_offset=0, other_info=None):
        #print("save_range", volume.shape, index_offset, vol_slice)
        if self.arr is not None:
            self.arr[vol_slice[0]-index_offset:vol_slice[1]-index_offset, :, :] = volume
        self.nwrites += volume.shape[0]