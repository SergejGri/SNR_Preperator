''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import time, datetime, traceback, sys, numpy as np, psutil

verbose = False
def vprint(*args):
    if verbose:
        print(*args)


def get_free_memory(units_exp=3):
    if get_free_memory.override is not None:
        return get_free_memory.override*1024**(3-units_exp)
    free_memory = max(psutil.virtual_memory()[1]/1024**units_exp - 0.5, 0.0)  # this is the currently free memory !, 0.5 GB safety margin
    if free_memory < 0.1:
        print('WARNING: computer is out of free memory')
    return min(free_memory*get_free_memory.use_fraction, get_free_memory.use_maximal*1024**(3-units_exp))
get_free_memory.use_fraction = 0.95
get_free_memory.use_maximal = np.inf # in GB
get_free_memory.override = None # in GB


def print_runtime(name='runtime', t0=0, min_time=0):
    t = time.time()-t0
    if t > min_time:
        minutes = t // 60
        seconds = t % 60
        if minutes == 0:
            if seconds < 1e-5:   print(name+': {:.3f} micro s'.format(seconds*1e6))
            elif seconds < 1e-2: print(name+': {:.3f} ms'.format(seconds*1e3))
            else:                print(name+': {:.3f} s'.format(seconds))
        else:
            print(name+': {:.0f} min, {:.1f} s'.format(minutes, seconds))
    return t


def get_exception_text_html():
    lines = (traceback.format_exception(*sys.exc_info()))
    text = '<p><b>Python traceback:</b>\n'
    for k, line in enumerate(lines):
        text += line
        if k == len(lines)-2:
            text += "<p><b>Error description:</b>\n"
    text = text.replace("\n", "\n<br>")
    text = text.replace("    ", "&nbsp;&nbsp;&nbsp;&nbsp;")
    return text


def get_exception_text():
    lines = (traceback.format_exception(*sys.exc_info()))
    text = ''
    for k, line in enumerate(lines):
        text += line
    return text


def undo_html(text):
    text = text.replace("<p>", "")
    text = text.replace("<b>", "")
    text = text.replace("</b>", "")
    text = text.replace("<br>", "")
    text = text.replace("&nbsp;", " ")
    return text


def nested_dict_update(source_dict, target_dict):
    for key in source_dict.keys():
        if type(source_dict[key]) is dict:
            try:
                target_dict[key]
            except KeyError:
                target_dict[key] = dict()
            finally:
                nested_dict_update(source_dict[key], target_dict[key])
        else:
            target_dict[key] = source_dict[key]


forbidden_modules = ('os.', 'sys.', 'shutil.')
def check_eval_string(eval_text):
    for forbidden_module in forbidden_modules:
        if forbidden_module in eval_text:
            raise ValueError(f'module {forbidden_module.rstrip(".")} may not be used in eval')
    if "import " in eval_text:
            raise ValueError('importing may not be used in eval')


class TimeFinishedEstimator:
    def __init__(self):
        self.timestamps = None
        self.estimated_speed = None
        self.k_first = None
        self.k_now = None

    def start(self, number_items):
        self.timestamps = np.zeros(number_items, 'f8')
        self.estimated_speed = None
        self.k_first = np.inf
        self.k_now = 0

    def reset(self):
        self.timestamps = None
        self.estimated_speed = None
        self.k_first = None
        self.k_now = None

    def item_done(self, k):
        if k < len(self.timestamps):
            self.k_first = min(k, self.k_first)
            self.timestamps[k] = time.time()

            if k > 1:
                past_samples = np.linspace(self.k_first, k, int(5 * (1 - np.exp(-0.05 * k)) + 2), dtype='u8', endpoint=True)
                speed_estimates = np.zeros(len(past_samples)-1, 'f8')
                for k_loc, (k_last, k_next) in enumerate(zip(past_samples[:-1], past_samples[1:])):
                    if k_next > k_last:
                        speed_estimates[k_loc] = (self.timestamps[k_next] - self.timestamps[k_last])/(k_next - k_last)

                estimated_speed = np.nanmedian(speed_estimates)
                update_significant = self.estimated_speed is None or not (self.estimated_speed*0.97 < estimated_speed < self.estimated_speed*1.03)
                if np.isfinite(estimated_speed) and update_significant:
                    self.estimated_speed = estimated_speed
                self.k_now = k

    def job_runtime_str(self):
        return self.convert_seconds_to_hms(self.timestamps[self.k_now]-self.timestamps[self.k_first])

    def item_runtime(self):
        return (self.timestamps[self.k_now]-self.timestamps[self.k_first])/(self.k_now+1)

    @staticmethod
    def convert_seconds_to_hms(time_s):
        if time_s > 3600:
            return f'{time_s//3600:.0f}:{(time_s % 3600)//60:02.0f}:{time_s % 60:02.0f}'
        elif time_s > 60:
            return f'{(time_s % 3600)//60:.0f}:{time_s % 60:02.0f}'
        else:
            return f'{time_s:.0f} s'

    def predict_finished_time(self, prepend='expected end: '):
        if self.estimated_speed is not None:
            time_s = (len(self.timestamps) - self.k_now) * self.estimated_speed + self.timestamps[self.k_now]
            localtime = time.localtime(time_s)
            clock = '{:02}:{:02}'.format(localtime.tm_hour, localtime.tm_min)
            diff_str = self.convert_seconds_to_hms(time_s - time.time())
            return prepend + clock + f' ({diff_str} left)'
        else:
            return '?'


