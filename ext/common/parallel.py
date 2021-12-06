''' Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''


import threading, multiprocessing, queue, psutil, cProfile, gc
from common import misc


class QueueThread(threading.Thread):
    def __init__(self, maxsize=10, error_call=None):
        super().__init__(daemon=True)
        self.cmd_queue = queue.Queue(maxsize=maxsize)
        self.stop = False
        self.result = None
        self.error_call = error_call
        if self.error_call is None:
            self.error_call = lambda t, b: print(misc.undo_html(t))

        self.result_ready_event = threading.Event()

        self.start()

    def put(self, func, *args):
        assert not self.stop, "QueueTread is stopped"
        self.cmd_queue.put((func, args))

    def do_stop(self):
        if self.is_alive():
            self.stop = True
            self.join()

    def wait(self):
        self.cmd_queue.join()

    def finish(self):
        if self.is_alive():
            self.cmd_queue.put(("stop", None))
            self.join()

    def has_result(self):
        return self.result_ready_event.is_set()

    def run(self):
        while not self.stop:
            try:
                func, args = self.cmd_queue.get()
                self.result_ready_event.clear()
                #print("queue length", self.cmd_queue.qsize(), func, args)
                if func == "stop":
                    self.cmd_queue.task_done()
                    return
                self.result = func(*args)
                self.cmd_queue.task_done()
                self.result_ready_event.set()
                gc.collect()
            except Exception as exc:
                self.error_call(misc.get_exception_text_html(), True)


class QueueProcessPool():
    def __init__(self, nworkers=None, use_cProfile=False):
        super().__init__()
        if nworkers is None:
            nworkers = psutil.cpu_count(logical=False)
        self.nworkers = nworkers
        self.use_cProfile = use_cProfile
        self.cmd_queue = multiprocessing.Queue()
        self.ret_queue = multiprocessing.Queue()
        self.workers = [QueueProcess(k, self.cmd_queue, self.ret_queue, use_cProfile) for k in range(nworkers)]

        self.is_running = True


    def put(self, func, *args):
        self.cmd_queue.put((func, args))

    def get_result(self, wait=True):
        return self.ret_queue.get(block=wait)

    def get_results(self):
        results = []
        while True:
            try:
                results.append(self.get_result(wait=False))
            except queue.Empty:
                break
        return results

    def do_stop(self):
        queue_empty = self.cmd_queue.empty()
        while not queue_empty:
            try:
                self.cmd_queue.get(block=False)
            except queue.Empty:
                queue_empty = True
        self.__exit__()

    def __del__(self):
        self.__exit__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type=None, exc_val=None, exc_tb=None):
        if self.is_running:
            print("exiting QueueProcessPool")
            [self.cmd_queue.put(("stop", None)) for worker in self.workers]
            [worker.join() for worker in self.workers]
            del self.workers[:]
            self.is_running = False
            print("exited QueueProcessPool")
        return exc_type is None



class QueueProcess(multiprocessing.Process):
    def __init__(self, worker_index, emd_queue, ret_queue, use_cProfile):
        super().__init__()
        self.daemon = True
        self.worker_index = worker_index
        self.cmd_queue = emd_queue
        self.ret_queue = ret_queue
        self.use_cProfile = use_cProfile
        self.stop = False
        self.start()

    def run(self):
        while not self.stop:
            func, args = self.cmd_queue.get()
            if func == "stop":
                self.stop = True
            else:
                if self.use_cProfile and self.worker_index == 0:
                    profile = cProfile.Profile()
                    result = profile.runcall(func, *args)
                    profile.print_stats()
                else:
                    result = func(*args)
                self.ret_queue.put(result)
        print("exiting worker", self.worker_index)