''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
# Developed at the Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany
import os, numpy as np, fnmatch, glob
pjoin = os.path.join


def sort_numbered(files_list, sort_name_range=(None, None)):
    file_unsorted_indices = np.arange(len(files_list))
    file_sorted_indices = np.zeros(len(files_list), dtype='i4')-1
    for number_length in np.arange(1, 6)[::-1]:
        for k in file_unsorted_indices[file_sorted_indices < 0]:
            fname = files_list[k][sort_name_range[0]:sort_name_range[1]]
            if file_sorted_indices[k] < 0:
                parts = fname.split('.')
                if len(parts) > 1 :
                    bare_fname = parts[-2]
                else:
                    bare_fname = parts[0]
                try:
                    n = int(bare_fname[-number_length:])
                    file_sorted_indices[k] = n
                except ValueError:
                    pass
    if np.all(file_sorted_indices >= 0):
        files_list_sorted = []
        sorters = np.argsort(file_sorted_indices)
        for k in sorters:
            files_list_sorted.append(files_list[k])
        return files_list_sorted
    else:
        return sorted(files_list)
    
    
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


def modify_path(path, suffix=None, before='', after=''):
    folder, filename = os.path.split(path)
    stem, old_suffix = separate_suffix(filename)
    if suffix is None:
        suffix = old_suffix
    suffix_append = '.' + suffix if suffix != '' else ''
    return os.path.normpath(pjoin(folder, before + stem + after + suffix_append))


def assure_dir_exists(path):
    if path != '':
        try:                    os.makedirs(path, exist_ok=True)
        except FileExistsError: pass


def do_exclude_patterns(filenames, exclude_patterns):
    for pattern in exclude_patterns:
        match_list = fnmatch.filter(filenames, '*'+pattern)
        if len(match_list) > 0:
            match_indices = []
            for match in match_list:
                match_indices.append(filenames.index(match))
            match_indices.sort()
            for k in match_indices[::-1]:
                del filenames[k]


def try_delete_file(filename):
    try:
        os.remove(filename)
        return True
    except OSError:
        return False

def try_delete_array_file(filename):
    for filename_ in (filename, separate_suffix(filename)[0] + '_info.txt',
                      separate_suffix(filename)[0] + '_info.pydict'):
        try:
            if os.path.isfile(filename_):
                os.remove(filename_)
            return True
        except OSError as err:
            print('file could not be deleted:', filename_, err)
            return False