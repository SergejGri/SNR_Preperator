'''written by Maximilian Ullherr, maximilian.ullherr@physik.uni-wuerzburg.de, Lehrstuhl fuer Roentgenmikroskopie/Universitaet Wuerzburg, Josef-Martin-Weg 63, 97074 Wuerzburg, Germany

This is an API translation layer between Python and OpenCl, mainly written for executing kernels

Currently, only part of the OpenCl functionality is implemented (only which was needed for this project)

Notable features:
 - Easy to use and Pythonic -- simple objects, kernel kwargs and type checking
 - Effectively no dependencies, very easy installation (and maintenance)
 - Helpful error messages e.g. on wrong kernel arguments, optional dtype checking for arguments (without dtype checking,
   e.g. a float64 can be used as a int2 vector or a uint64 as float64 because OpenCl only checks length)
 - The python side uses numpy array shape axis order (zyx) to avoid confusion
   (for image creation and work sizes, but not for vectors passed to kernels)
 - Simplifies some OpenCl internals into easier to use objects (e.g. ComputeDevice) without sacrificing the API capabilities
 - Automatic cleanup of OpenCl memory objects on deletion of the Buffer/Image in Python (also called RAII)

Main functions:
 - get_device_infos:    list information about the available devices
 - get_compute_device:  initialize an OpenCl computation environment (also has an ..._interactive variant)

Main classes:
 - ComputeDevice:       OpenCl device object for kernel execution and memory object management (contains context+queue)
 - Program:             Program (collection of compiled kernels as Kernel objects)
 - Kernel:              Kernel object used to execute a OpenCl kernel (created automatically), see Kernel.__call__()
 - Buffer:              Data array on the device
 - Image:               Image on the device
Whenever reasonable, class methods are used to provide functionality.

Other functions:
 - unload_lib:          shut down API layer and free resources

Note: On import of this module, the API layer is not loaded. This happens at the first use (device creation)
and if OpenCl is not available on this machine, an exception is raised at this point.

As this module is written completely in Python, speed may be lower than other implementations.
We have found no cases where there is a significant performance difference between this module and e.g. pyopencl.
One kernel execution has an overhead of approximately 50-100 microseconds (including OpenCl API calls)

Note
 - This module only works on 64 bit platforms

Planned functionality for the future:
 - caching of program binaries
 - support for more API functionality

License for this code:
Copyright 2015-2020 University Wuerzburg.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import ctypes, numpy as np, string, sys, platform, hashlib, os
from ctypes.util import find_library

possible_library_names = ('OpenCL.dll', 'libOpenCL.so.1', 'libOpenCL.so')  # you can edit this if the OpenCl library cannot be found
cl_lib = None  # ctypes.cdll instance that is created on first use of a module function (clGetPlatformIDs), see load_lib() and unload_lib()
version = 1, 0
version_str = f'v{".".join([str(v) for v in version])}'

if platform.architecture()[0] != '64bit':
    print('WARNING: detected a non-64bit platform architecture, the opencl module may not work.')

CHECK_KERNEL_ARGS = False   # if this is enabled, kernel args are always checked for correct type
                            # reduces performance of the kernel call by a few microseconds per argument,
                            # but helps to avoid dtype errors (e.g. int32 instead of float32)
                            # wrong vector lengths or bit widths (e.g. int16 instead of int32) are detected anyway
                            # it is  STRONGLY RECOMMENDED to enable this during development (but not in use)
                            # this option will enable itself if a debugger is used (see below)
if sys.gettrace() is not None and 'pdb' in sys.modules:  # enables argument checking if a debugger is used
    CHECK_KERNEL_ARGS = True
    print('found debugger in use, enabling opencl kernel arg checking')

PRINT_SOURCE_ON_BUILD_FAIL = True  # print the whole program source on a build failure (with line numbers)
VERBOSE = False  # print some information about everything that happens
TEST_CLIP_MAX_GLOBAL_MEMORY_PROPERTY = None  # limits DeviceInfo.global_mem_size to this value for testing purposes



# ===== constants and error classes =====
class Constants:
    # these are all constants not part of another class (e.g. ComputeDevice.ImInfo)
    VERSION_1_0 = 1
    VERSION_1_1 = 1
    VERSION_1_2 = 1
    VERSION_2_0 = 1
    VERSION_2_1 = 1
    VERSION_2_2 = 1
    FALSE = 0
    TRUE = 1
    BLOCKING =TRUE
    NON_BLOCKING =FALSE

    class PlatformInfo:
       PLATFORM_PROFILE = 0x0900
       PLATFORM_VERSION = 0x0901
       PLATFORM_NAME = 0x0902
       PLATFORM_VENDOR = 0x0903
       PLATFORM_EXTENSIONS = 0x0904
       PLATFORM_HOST_TIMER_RESOLUTION = 0x0905

    class DeviceFPConfig:
       FP_DENORM = (1<<0)
       FP_INF_NAN = (1<<1)
       FP_ROUND_TO_NEAREST = (1<<2)
       FP_ROUND_TO_ZERO = (1<<3)
       FP_ROUND_TO_INF = (1<<4)
       FP_FMA = (1<<5)
       FP_SOFT_FLOAT = (1<<6)
       FP_CORRECTLY_ROUNDED_DIVIDE_SQRT = (1<<7)

    class DeviceMemCacheType:
       NONE = 0x0
       READ_ONLY_CACHE = 0x1
       READ_WRITE_CACHE = 0x2

    class DeviceLocalMemType:
       LOCAL = 0x1
       GLOBAL = 0x2

    class DeviceExecCapabilities:
       EXEC_KERNEL = (1<<0)
       EXEC_NATIVE_KERNEL = (1<<1)

    class CommandQueueProperties:
       QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE = (1<<0)
       QUEUE_PROFILING_ENABLE = (1<<1)
       QUEUE_ON_DEVICE = (1<<2)
       QUEUE_ON_DEVICE_DEFAULT = (1<<3)

    class ContextInfo:
       CONTEXT_REFERENCE_COUNT = 0x1080
       CONTEXT_DEVICES = 0x1081
       CONTEXT_PROPERTIES = 0x1082
       CONTEXT_NUM_DEVICES = 0x1083

    class ContextProperties:
       CONTEXT_PLATFORM = 0x1084
       CONTEXT_INTEROP_USER_SYNC = 0x1085

    class CommandQueueInfo:
       QUEUE_CONTEXT = 0x1090
       QUEUE_DEVICE = 0x1091
       QUEUE_REFERENCE_COUNT = 0x1092
       QUEUE_PROPERTIES = 0x1093
       QUEUE_SIZE = 0x1094
       QUEUE_DEVICE_DEFAULT = 0x1095

    class PipeInfo:
       PIPE_PACKET_SIZE = 0x1120
       PIPE_MAX_PACKETS = 0x1121

    class MapFlags:
       MAP_READ = (1<<0)
       MAP_WRITE = (1<<1)
       MAP_WRITE_INVALIDATE_REGION = (1<<2)

    class EventInfo:
       EVENT_COMMAND_QUEUE = 0x11D0
       EVENT_COMMAND_TYPE = 0x11D1
       EVENT_REFERENCE_COUNT = 0x11D2
       EVENT_COMMAND_EXECUTION_STATUS = 0x11D3
       EVENT_CONTEXT = 0x11D4

    class CommandType:
       COMMAND_NDRANGE_KERNEL = 0x11F0
       COMMAND_TASK = 0x11F1
       COMMAND_NATIVE_KERNEL = 0x11F2
       COMMAND_READ_BUFFER = 0x11F3
       COMMAND_WRITE_BUFFER = 0x11F4
       COMMAND_COPY_BUFFER = 0x11F5
       COMMAND_READ_IMAGE = 0x11F6
       COMMAND_WRITE_IMAGE = 0x11F7
       COMMAND_COPY_IMAGE = 0x11F8
       COMMAND_COPY_IMAGE_TO_BUFFER = 0x11F9
       COMMAND_COPY_BUFFER_TO_IMAGE = 0x11FA
       COMMAND_MAP_BUFFER = 0x11FB
       COMMAND_MAP_IMAGE = 0x11FC
       COMMAND_UNMAP_MEM_OBJECT = 0x11FD
       COMMAND_MARKER = 0x11FE
       COMMAND_ACQUIRE_GL_OBJECTS = 0x11FF
       COMMAND_RELEASE_GL_OBJECTS = 0x1200
       COMMAND_READ_BUFFER_RECT = 0x1201
       COMMAND_WRITE_BUFFER_RECT = 0x1202
       COMMAND_COPY_BUFFER_RECT = 0x1203
       COMMAND_USER = 0x1204
       COMMAND_BARRIER = 0x1205
       COMMAND_MIGRATE_MEM_OBJECTS = 0x1206
       COMMAND_FILL_BUFFER = 0x1207
       COMMAND_FILL_IMAGE = 0x1208
       COMMAND_SVM_FREE = 0x1209
       COMMAND_SVM_MEMCPY = 0x120A
       COMMAND_SVM_MEMFILL = 0x120B
       COMMAND_SVM_MAP = 0x120C
       COMMAND_SVM_UNMAP = 0x120D

    class CommandExecutionStatus:
       COMPLETE = 0x0
       RUNNING = 0x1
       SUBMITTED = 0x2
       QUEUED = 0x3

    class ProfilingInfo:
       PROFILING_COMMAND_QUEUED = 0x1280
       PROFILING_COMMAND_SUBMIT = 0x1281
       PROFILING_COMMAND_START = 0x1282
       PROFILING_COMMAND_END = 0x1283
       PROFILING_COMMAND_COMPLETE = 0x1284


error_codes = {0: "CL_SUCCESS",-1: "CL_DEVICE_NOT_FOUND",-2: "CL_DEVICE_NOT_AVAILABLE",-3: "CL_COMPILER_NOT_AVAILABLE",-4: "CL_MEM_OBJECT_ALLOCATION_FAILURE",-5: "CL_OUT_OF_RESOURCES",-6: "CL_OUT_OF_HOST_MEMORY",-7: "CL_PROFILING_INFO_NOT_AVAILABLE",-8: "CL_MEM_COPY_OVERLAP",-9: "CL_IMAGE_FORMAT_MISMATCH",-10: "CL_IMAGE_FORMAT_NOT_SUPPORTED",-11: "CL_BUILD_PROGRAM_FAILURE",-12: "CL_MAP_FAILURE",-13: "CL_MISALIGNED_SUB_BUFFER_OFFSET",-14: "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",-15: "CL_COMPILE_PROGRAM_FAILURE",-16: "CL_LINKER_NOT_AVAILABLE",-17: "CL_LINK_PROGRAM_FAILURE",-18: "CL_DEVICE_PARTITION_FAILED",-19: "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",-30: "CL_INVALID_VALUE",-31: "CL_INVALID_DEVICE_TYPE",-32: "CL_INVALID_PLATFORM",-33: "CL_INVALID_DEVICE",-34: "CL_INVALID_CONTEXT",-35: "CL_INVALID_QUEUE_PROPERTIES",-36: "CL_INVALID_COMMAND_QUEUE",-37: "CL_INVALID_HOST_PTR",-38: "CL_INVALID_MEM_OBJECT",-39: "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",-40: "CL_INVALID_IMAGE_SIZE",-41: "CL_INVALID_SAMPLER",-42: "CL_INVALID_BINARY",-43: "CL_INVALID_BUILD_OPTIONS",-44: "CL_INVALID_PROGRAM",-45: "CL_INVALID_PROGRAM_EXECUTABLE",-46: "CL_INVALID_KERNEL_NAME",-47: "CL_INVALID_KERNEL_DEFINITION",-48: "CL_INVALID_KERNEL",-49: "CL_INVALID_ARG_INDEX",-50: "CL_INVALID_ARG_VALUE",-51: "CL_INVALID_ARG_SIZE",-52: "CL_INVALID_KERNEL_ARGS",-53: "CL_INVALID_WORK_DIMENSION",-54: "CL_INVALID_WORK_GROUP_SIZE",-55: "CL_INVALID_WORK_ITEM_SIZE",-56: "CL_INVALID_GLOBAL_OFFSET",-57: "CL_INVALID_EVENT_WAIT_LIST",-58: "CL_INVALID_EVENT",-59: "CL_INVALID_OPERATION",-60: "CL_INVALID_GL_OBJECT",-61: "CL_INVALID_BUFFER_SIZE",-62: "CL_INVALID_MIP_LEVEL",-63: "CL_INVALID_GLOBAL_WORK_SIZE",-64: "CL_INVALID_PROPERTY",-65: "CL_INVALID_IMAGE_DESCRIPTOR",-66: "CL_INVALID_COMPILER_OPTIONS",-67: "CL_INVALID_LINKER_OPTIONS",-68: "CL_INVALID_DEVICE_PARTITION_COUNT",-69: "CL_INVALID_PIPE_SIZE",-70: "CL_INVALID_DEVICE_QUEUE",-71: "CL_INVALID_SPEC_ID",-72: "CL_MAX_SIZE_RESTRICTION_EXCEEDED", 1: 'OpenCl library function was not called or call failed'}


cl_kernel_argtypes = {'float': 'float32', 'double': 'float64', 'uchar': 'uint8', 'char': 'int8',
    'ushort': 'uint16', 'short': 'int16', 'uint': 'uint32', 'int': 'int32', 'ulong': 'uint64', 'long': 'int64'}

cl_numpy_dtype_str = {'f4': 'float', 'f8': 'double', 'u1': 'uchar', 'i1': 'char',
    'u2': 'ushort', 'i2': 'short', 'u4': 'uint', 'i4': 'int', 'u8': 'ulong', 'i8': 'long'}


class Error(Exception):
    pass


class DeviceMemoryError(Error):
    pass


class LibError(Error):
    def __init__(self, index, descr=''):
        self.index = index
        self.descr = descr

    def __str__(self):
        if self.index == -5:
            self.descr += '(an error occurred in OpenCl before this call but cannot be located)'
        try:
            return f'OpenClError: {error_codes[self.index]} {self.descr}'
        except KeyError:
            return f'OpenClError: unknown library error code {self.index} {self.descr}'


class CodeError(Error):
    pass


# ===== loading and calling the library =====
def load_lib(possible_library_names=possible_library_names):
    '''
    load the OpenCl library
    automatically called in clGetPlatformIDs()

    uses ctypes.util.find_library() to locate the OpenCl library

    :param possible_library_names:  additional list of possible names of the OpenCl library file
    :return:
    '''
    cl_lib_imported, err_msg = None, ''
    try:
        lib_name = find_library('OpenCL')
        if lib_name is not None:
            cl_lib_imported = ctypes.cdll.LoadLibrary(lib_name)
            err_msg += f'loaded library "{lib_name}"'
        else:
            raise OSError('find_library failed')
    except OSError as err:
        err_msg = repr(err)

    if cl_lib_imported is None or cl_lib_imported._name is None:
        for name in possible_library_names:
            try:
                cl_lib_imported = ctypes.cdll.LoadLibrary(name)
                err_msg += f'loaded library "{name}"'
                break
            except OSError as err:
                err_msg += '\n'+repr(err)

    if cl_lib_imported is None:
        raise Error(f'Failed to load OpenCl library, is OpenCl installed on this computer?\n{err_msg}')

    if VERBOSE:
        print(err_msg)

    global cl_lib # global statements cannot be used in for loops
    cl_lib = cl_lib_imported


def unload_lib():
    '''
    free the resources allocated by the program (e.g. approximately 100 MB of GPU RAM)

    Note:   Any objects of the types Buffer, Image or Program become invalid with this call.
            Using them afterwards will lead to an interpreter crash (OpenCl library tries to use an invalid memory address)
            It is recommended to only use this function when no Buffer, Image or Program objects were created at all
    :return:
    '''
    global cl_lib
    cl_lib = None


def call_dll(function, *args):
    # call to the OpenCl dll with standard error code returns
    err_code = function(*args)
    if err_code != 0:
        if err_code == -4:
            raise DeviceMemoryError(f'OpenCl failed to allocate device memory, buffer/image too large for device memory? ({function.__name__})')
        else:
            raise LibError(err_code)


def call_dll_ret(function, *args):
    # call to the OpenCl dll when return is a identifier (e.g. context, program, memobject, ...) of type uint64
    function.restype = ctypes.c_uint64
    errorcode_ret = ctypes.c_int32(1)
    return_val = function(*args, ctypes.byref(errorcode_ret))
    if errorcode_ret.value != 0:
        raise LibError(errorcode_ret.value)
    return ctypes.c_uint64(return_val)


# ===== directly translated OpenCl library functions =====
def clGetPlatformIDs():
    if cl_lib is None:
            load_lib()
    platforms = (ctypes.c_uint64 * 64)()
    num_platforms = ctypes.c_uint64()
    try:
        call_dll(cl_lib.clGetPlatformIDs, 64, ctypes.byref(platforms), ctypes.byref(num_platforms))
    except AttributeError:
        raise Error(f'Invalid OpenCl library loaded.')
    return [ctypes.c_uint64(p) for p in platforms[:num_platforms.value]]


def clGetDeviceIDs(platform_id, device_type='all'):
    if device_type == 'all':
        device_type = DeviceInfo.Type.DEVICE_TYPE_ALL
    devices = (ctypes.c_uint64 * 64)()
    num_devices = ctypes.c_uint64()
    try:
        call_dll(cl_lib.clGetDeviceIDs, platform_id, device_type,
                 64, ctypes.byref(devices), ctypes.byref(num_devices))
        return [ctypes.c_uint64(d) for d in devices[:num_devices.value]]
    except LibError:
        return []


def print_error(errinfo, private_info, ch, user_data):
    k = 0
    err_str = b''
    while True:
        char = errinfo[k]
        if char == b'\0':
            break
        else:
            err_str += char
    print('OpenCl error at runtime:', str(err_str, 'ascii'), '\n', str(private_info[:ch.value], 'ascii'))


pfn_notify_functype = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char),
                                       ctypes.c_uint64, ctypes.POINTER(ctypes.c_char))
pfn_notify = pfn_notify_functype(print_error)  # this does not appear to work (or was not called from lib during tests)


def clCreateContext(platform_id, device_id):
    # note: cl_uint is uint32
    properties_c = (ctypes.c_uint64 * 3)(Constants.ContextProperties.CONTEXT_PLATFORM, platform_id, 0)
    #print(properties_c[:])
    device_id_c = (ctypes.c_uint64 * 1)(device_id)
    #print(device_id_c[:])
    return call_dll_ret(cl_lib.clCreateContext, ctypes.byref(properties_c), ctypes.c_uint32(1),
                        ctypes.byref(device_id_c),
                        None, None)


def clCreateCommandQueue(cl_context, cl_device_id):
    properties_c = ctypes.c_uint64(0)
    return call_dll_ret(cl_lib.clCreateCommandQueue, cl_context, cl_device_id, properties_c)


# ===== select/create a computing device =====
def get_device_infos(device_type='auto'):
    '''
    Get the available devices on this computer.

    :param device_type: One of DEVICE_TYPE_ALL,DEVICE_TYPE_GPU,DEVICE_TYPE_CPU,DEVICE_TYPE_ACCELERATOR
                        The default 'auto' gives all GPU and CPU devices (in this order such that devices[0] is a GPU if available)
                        A custom sorting is applied for the following order: GPUs, CPUs, Intel GPUs, other
                        (Intel GPUs are sorted to the end because they do not support float images)
    :return:            list of DeviceInfo instances that can be used to initialize ComputeDevice
    '''
    platforms = clGetPlatformIDs()
    device_ids = []
    if device_type == 'auto':
        for platform in platforms:
            device_ids += clGetDeviceIDs(platform, DeviceInfo.Type.DEVICE_TYPE_GPU)
        for platform in platforms:
            device_ids += clGetDeviceIDs(platform, DeviceInfo.Type.DEVICE_TYPE_CPU)
    else:
        for platform in platforms:
            device_ids += clGetDeviceIDs(platform, device_type)
    devices = [DeviceInfo(device_id) for device_id in device_ids]

    devices.sort(key=device_sort)

    if VERBOSE:
        print('==== get_devices() found the following OpenCl devices =====', *devices, '====', sep='\n')
    if len(devices) == 0:
        print('WARNING: no opencl devices available - either (1) no appropriate drivers are installed or '
              '(2) this process was forked (in which case OpenCl does not work at all)')
    return devices


def get_device_info(index=0, device_type='auto'):
    '''
        Get a DeviceInfo identified by an index (e.g. for use in combination with device selection using a QComboBox)

        Calling without arguments will select an arbitrary device (a GPU if available)

        :param index:       Index of the device as in the list returned by get_devices() with the same device_type

        :param device_type: One of DEVICE_TYPE_ALL,DEVICE_TYPE_GPU,DEVICE_TYPE_CPU,DEVICE_TYPE_ACCELERATOR
                            The default 'auto' gives all GPU and CPU devices (in this order such that devices[0] is a GPU if available)
        :return:
        '''
    devices = get_device_infos(device_type)
    return devices[index]


def get_compute_device(index=0, device_type='auto'):
    '''
    Get a ComputeDevice identified by an index (e.g. for use in combination with device selection using a QComboBox)

    Calling without arguments will select an arbitrary device (a GPU if available)

    :param index:       Index of the device as in the list returned by get_devices() with the same device_type

    :param device_type: One of DEVICE_TYPE_ALL,DEVICE_TYPE_GPU,DEVICE_TYPE_CPU,DEVICE_TYPE_ACCELERATOR
                        The default 'auto' gives all GPU and CPU devices (in this order such that devices[0] is a GPU if available)
    :return:
    '''
    device = ComputeDevice(get_device_info(index, device_type))
    print(f'selected OpenCL compute device: {device}')
    return device


def get_compute_device_interactive(default_index=0, device_type='auto'):
    '''
    Get a device when using an interactive interpreter or a Jupyter Notebook

    :param default_index:  Default index to use of the device as in the list returned by get_devices() with the same device_type

    :param device_type: One of DeviceInfo.Type.*
                        The default 'auto' gives all GPU and CPU devices (in this order such that devices[0] is a GPU if available)
    :return:
    '''
    devices = get_device_infos(device_type)
    answer, index = None, default_index

    while True:
        if answer is None:
            ask_msg = 'Select OpenCl device:'
            for k, device_info in enumerate(devices):
                ask_msg += f'\n [{k+1}] {device_info.short_name()}'
            ask_msg += f'\nDefault: {devices[default_index].short_name()} \n'
        else:
            ask_msg = ""
        answer = input(ask_msg)
        if answer == '':
            break
        try:
            n = int(answer)
            if 0 < n <= len(devices):
                index = n - 1
                break
            elif n == 0:
                raise TypeError("canceled execution for input==0")
            else:
                print("invalid index:", n, "try again")

        except ValueError:
            print('invalid input')

    print(f'selected {devices[index].short_name()}')
    device = ComputeDevice(devices[index])
    print(f'selected OpenCL compute device: {device}')
    return device


class DeviceInfo:
    def __init__(self, device_id: ctypes.c_uint64):
        '''
        An object representing an available OpenCl device

        Note: Instances of this object cannot be transferred between different imports of this module
        (e.g. different processes) and become invalid after use of unload_lib()

        :param device_id:   ID of the device in the OpenCl lib
        '''
        self.id = device_id
        self.name = self.get_info(self.Info.DEVICE_NAME, str)
        self.vendor = self.get_info(self.Info.DEVICE_VENDOR, str)
        self.cl_version = self.get_info(self.Info.DEVICE_VERSION, str)
        self.driver_version = self.get_info(self.Info.DRIVER_VERSION, str)
        self.global_mem_size = np.min(self.get_info(self.Info.DEVICE_GLOBAL_MEM_SIZE, int), TEST_CLIP_MAX_GLOBAL_MEMORY_PROPERTY)
        self.type = self.device_types.get(self.get_info(self.Info.DEVICE_TYPE, int), 'unknown')
        self.max_work_item_sizes = self.get_info(self.Info.DEVICE_MAX_WORK_ITEM_SIZES)
        self.max_work_group_size = self.get_info(self.Info.DEVICE_MAX_WORK_GROUP_SIZE, int)

    def __str__(self):
        return f'{self.cl_version} device {self.vendor} {self.name} {self.type}, ' \
               f'v{self.driver_version}, {self.global_mem_size/1024**3:.3f} GB memory (id {self.id.value})'

    class Info:
       DEVICE_TYPE = 0x1000
       DEVICE_VENDOR_ID = 0x1001
       DEVICE_MAX_COMPUTE_UNITS = 0x1002
       DEVICE_MAX_WORK_ITEM_DIMENSIONS = 0x1003
       DEVICE_MAX_WORK_GROUP_SIZE = 0x1004
       DEVICE_MAX_WORK_ITEM_SIZES = 0x1005
       DEVICE_PREFERRED_VECTOR_WIDTH_CHAR = 0x1006
       DEVICE_PREFERRED_VECTOR_WIDTH_SHORT = 0x1007
       DEVICE_PREFERRED_VECTOR_WIDTH_INT = 0x1008
       DEVICE_PREFERRED_VECTOR_WIDTH_LONG = 0x1009
       DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT = 0x100A
       DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE = 0x100B
       DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
       DEVICE_ADDRESS_BITS = 0x100D
       DEVICE_MAX_READ_IMAGE_ARGS = 0x100E
       DEVICE_MAX_WRITE_IMAGE_ARGS = 0x100F
       DEVICE_MAX_MEM_ALLOC_SIZE = 0x1010
       DEVICE_IMAGE2D_MAX_WIDTH = 0x1011
       DEVICE_IMAGE2D_MAX_HEIGHT = 0x1012
       DEVICE_IMAGE3D_MAX_WIDTH = 0x1013
       DEVICE_IMAGE3D_MAX_HEIGHT = 0x1014
       DEVICE_IMAGE3D_MAX_DEPTH = 0x1015
       DEVICE_IMAGE_SUPPORT = 0x1016
       DEVICE_MAX_PARAMETER_SIZE = 0x1017
       DEVICE_MAX_SAMPLERS = 0x1018
       DEVICE_MEM_BASE_ADDR_ALIGN = 0x1019
       DEVICE_MIN_DATA_TYPE_ALIGN_SIZE = 0x101A
       DEVICE_SINGLE_FP_CONFIG = 0x101B
       DEVICE_GLOBAL_MEM_CACHE_TYPE = 0x101C
       DEVICE_GLOBAL_MEM_CACHELINE_SIZE = 0x101D
       DEVICE_GLOBAL_MEM_CACHE_SIZE = 0x101E
       DEVICE_GLOBAL_MEM_SIZE = 0x101F
       DEVICE_MAX_CONSTANT_BUFFER_SIZE = 0x1020
       DEVICE_MAX_CONSTANT_ARGS = 0x1021
       DEVICE_LOCAL_MEM_TYPE = 0x1022
       DEVICE_LOCAL_MEM_SIZE = 0x1023
       DEVICE_ERROR_CORRECTION_SUPPORT = 0x1024
       DEVICE_PROFILING_TIMER_RESOLUTION = 0x1025
       DEVICE_ENDIAN_LITTLE = 0x1026
       DEVICE_AVAILABLE = 0x1027
       DEVICE_COMPILER_AVAILABLE = 0x1028
       DEVICE_EXECUTION_CAPABILITIES = 0x1029
       DEVICE_QUEUE_PROPERTIES = 0x102A
       DEVICE_QUEUE_ON_HOST_PROPERTIES = 0x102A
       DEVICE_NAME = 0x102B
       DEVICE_VENDOR = 0x102C
       DRIVER_VERSION = 0x102D
       DEVICE_PROFILE = 0x102E
       DEVICE_VERSION = 0x102F
       DEVICE_EXTENSIONS = 0x1030
       DEVICE_PLATFORM = 0x1031
       DEVICE_DOUBLE_FP_CONFIG = 0x1032
       DEVICE_HALF_FP_CONFIG = 0x1033
       DEVICE_PREFERRED_VECTOR_WIDTH_HALF = 0x1034
       DEVICE_HOST_UNIFIED_MEMORY = 0x1035
       DEVICE_NATIVE_VECTOR_WIDTH_CHAR = 0x1036
       DEVICE_NATIVE_VECTOR_WIDTH_SHORT = 0x1037
       DEVICE_NATIVE_VECTOR_WIDTH_INT = 0x1038
       DEVICE_NATIVE_VECTOR_WIDTH_LONG = 0x1039
       DEVICE_NATIVE_VECTOR_WIDTH_FLOAT = 0x103A
       DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE = 0x103B
       DEVICE_NATIVE_VECTOR_WIDTH_HALF = 0x103C
       DEVICE_OPENCL_C_VERSION = 0x103D
       DEVICE_LINKER_AVAILABLE = 0x103E
       DEVICE_BUILT_IN_KERNELS = 0x103F
       DEVICE_IMAGE_MAX_BUFFER_SIZE = 0x1040
       DEVICE_IMAGE_MAX_ARRAY_SIZE = 0x1041
       DEVICE_PARENT_DEVICE = 0x1042
       DEVICE_PARTITION_MAX_SUB_DEVICES = 0x1043
       DEVICE_PARTITION_PROPERTIES = 0x1044
       DEVICE_PARTITION_AFFINITY_DOMAIN = 0x1045
       DEVICE_PARTITION_TYPE = 0x1046
       DEVICE_REFERENCE_COUNT = 0x1047
       DEVICE_PREFERRED_INTEROP_USER_SYNC = 0x1048
       DEVICE_PRINTF_BUFFER_SIZE = 0x1049
       DEVICE_IMAGE_PITCH_ALIGNMENT = 0x104A
       DEVICE_IMAGE_BASE_ADDRESS_ALIGNMENT = 0x104B
       DEVICE_MAX_READ_WRITE_IMAGE_ARGS = 0x104C
       DEVICE_MAX_GLOBAL_VARIABLE_SIZE = 0x104D
       DEVICE_QUEUE_ON_DEVICE_PROPERTIES = 0x104E
       DEVICE_QUEUE_ON_DEVICE_PREFERRED_SIZE = 0x104F
       DEVICE_QUEUE_ON_DEVICE_MAX_SIZE = 0x1050
       DEVICE_MAX_ON_DEVICE_QUEUES = 0x1051
       DEVICE_MAX_ON_DEVICE_EVENTS = 0x1052
       DEVICE_SVM_CAPABILITIES = 0x1053
       DEVICE_GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE = 0x1054
       DEVICE_MAX_PIPE_ARGS = 0x1055
       DEVICE_PIPE_MAX_ACTIVE_RESERVATIONS = 0x1056
       DEVICE_PIPE_MAX_PACKET_SIZE = 0x1057
       DEVICE_PREFERRED_PLATFORM_ATOMIC_ALIGNMENT = 0x1058
       DEVICE_PREFERRED_GLOBAL_ATOMIC_ALIGNMENT = 0x1059
       DEVICE_PREFERRED_LOCAL_ATOMIC_ALIGNMENT = 0x105A
       DEVICE_IL_VERSION = 0x105B
       DEVICE_MAX_NUM_SUB_GROUPS = 0x105C
       DEVICE_SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS = 0x105D

    class Type:
       DEVICE_TYPE_DEFAULT = (1 << 0)
       DEVICE_TYPE_CPU = (1 << 1)
       DEVICE_TYPE_GPU = (1 << 2)
       DEVICE_TYPE_ACCELERATOR = (1 << 3)
       DEVICE_TYPE_CUSTOM = (1 << 4)
       DEVICE_TYPE_ALL = 0xFFFFFFFF
       DEVICE_TYPE = 0x1000

    device_types = {Type.DEVICE_TYPE_CPU: 'CPU',
                    Type.DEVICE_TYPE_GPU: 'GPU',
                    Type.DEVICE_TYPE_ACCELERATOR: 'accelerator'}

    def get_info(self, cl_device_info, info_dtype=None):
        '''
        poll an info parameter

        :param cl_device_info:  must be an attribute of DeviceInfo.ImInfo
        :param info_dtype:      data type of the info
        :return:
        '''

        if cl_device_info is self.Info.DEVICE_MAX_WORK_ITEM_SIZES:
            info_value = (ctypes.c_uint64 * 3)()
            info_value_size = ctypes.c_uint64(24)
            info_size = ctypes.c_uint64()
            call_dll(cl_lib.clGetDeviceInfo, self.id,
                     cl_device_info, info_value_size, ctypes.byref(info_value), ctypes.byref(info_size))
            return info_value[:]

        elif info_dtype is str:
            info_str = ctypes.create_string_buffer(512)
            info_len = ctypes.c_uint64()
            call_dll(cl_lib.clGetDeviceInfo, self.id,
                     cl_device_info, 512, info_str, ctypes.byref(info_len))
            return str(info_str[:info_len.value - 1], 'ascii')

        elif info_dtype is int:
            info_value = ctypes.c_uint64()
            info_value_size = ctypes.c_uint64(8)
            info_size = ctypes.c_uint64()
            call_dll(cl_lib.clGetDeviceInfo, self.id,
                     cl_device_info, info_value_size, ctypes.byref(info_value), ctypes.byref(info_size))
            return info_value.value

        else:
            raise TypeError('must give info_dtype argument')

    def short_name(self):
        return f'{self.name} {self.type}'


def device_sort(device_info):
    if device_info.type == 'GPU' and 'Intel' in device_info.vendor:
        return 2
    elif device_info.type == 'GPU':
        return 0
    elif device_info.type == 'CPU':
        return 1
    else:
        return 3


class ComputeDevice:
    def __init__(self, device_info: DeviceInfo, context_id: ctypes.c_uint64=None):
        '''
        This is a computing device class for use in Program and Image/Buffer

        Note: Initiating a device will use up some memory (roughly 100 MB).
              This memory can only be freed by exiting the process, deleting the ComputeDevice does not work.

        It combines an OpenCl context, an OpenCl queue and the corresponding OpenCl device
        such that the ComputeDevice is a single object that can be used to handle memory objects
        and execute kernels.

        Multiple queues/contexts per device must use multiple ComputeDevice instances.
        Example:    device_1 = ComputeDevice(device_info_A)
                    device_2 = ComputeDevice(device_info_A, device_1.context_id)
        This way, device1 and device2 will share a context but have different queues.
        Notes:
        - Buffer/Image objects are valid context-wide.
        - Deleting device_1 will render device_2 unusable (the context is deinitialized, see self.owns_context)

        :param device_info: A DeviceInfo instance as returned by get_devices()
        :param context_id:  An optional context id created beforehand.
                            Use only if multiple queues per context are needed.
        '''
        self.info = device_info
        self.device_id = device_info.id
        self.platform_id = ctypes.c_uint64(self.info.get_info(DeviceInfo.Info.DEVICE_PLATFORM, int))
        if context_id is None:
            self.context_id: ctypes.c_uint64 = clCreateContext(self.platform_id, self.device_id)
            self.owns_context = True
        else:
            self.context_id: ctypes.c_uint64 = context_id
            self.owns_context = False
        self.queue_id = clCreateCommandQueue(self.context_id, self.device_id)
        self.allocated_mem = 0

        if VERBOSE:
            print('created', self)

    def __str__(self):
        return f'{self.__class__.__name__} {self.info} (context id {self.context_id.value}, queue id {self.queue_id.value})'

    def __del__(self):
        if hasattr(self, 'queue_id'):  # may get called before initialization is finished
            call_dll(cl_lib.clReleaseCommandQueue, self.queue_id)
        if hasattr(self, 'owns_context') and self.owns_context:
            call_dll(cl_lib.clReleaseContext, self.context_id)
    
    def get_buffer(self, mem_flags=None, size=None, arr: np.ndarray=None, copy_host_ptr='auto'):
        if mem_flags is None:
            mem_flags = Buffer.MemFlags.READ_WRITE
        return Buffer(self, mem_flags, size, arr, copy_host_ptr)

    def get_image(self, mem_flags=None, shape=None, format=None,
                 arr: np.ndarray=None, is_array=False, copy_host_ptr='auto'):
        if mem_flags is None:
            mem_flags = Image.MemFlags.READ_WRITE
        return Image(self, mem_flags, format, shape, arr, is_array, copy_host_ptr)


# ===== memory objects (images and buffers) =====
class MemObject:
    def __init__(self, device: ComputeDevice, cl_mem: ctypes.c_uint64):
        '''
        base class for memory objects (buffers, images) with automatic cleanup

        :param cl_mem: memory object identifier
        '''
        self.device = device
        self.cl_mem = cl_mem
        self.size = self.get_size()
        self.device.allocated_mem += self.size
        if VERBOSE:
            print(f'allocated {self.__class__.__name__} {self.cl_mem.value%2**32}, size {self.size/1024**3:.3f} GB, allocated {self.device.allocated_mem/1024**3:.3f} GB')

    def __del__(self):
        if hasattr(self, 'cl_mem'):
            call_dll(cl_lib.clReleaseMemObject, self.cl_mem)
            self.device.allocated_mem -= self.size
            if VERBOSE:
                print(f'cleared {self.__class__.__name__} {self.cl_mem.value%2**32}, size {self.size/1024**3:.3f} GB, allocated {self.device.allocated_mem/1024**3:.3f} GB')

    class MemFlags:
       READ_WRITE = (1 << 0)
       WRITE_ONLY = (1 << 1)
       READ_ONLY = (1 << 2)
       USE_HOST_PTR = (1 << 3)
       ALLOC_HOST_PTR = (1 << 4)
       COPY_HOST_PTR = (1 << 5)
       HOST_WRITE_ONLY = (1 << 7)
       HOST_READ_ONLY = (1 << 8)
       HOST_NO_ACCESS = (1 << 9)
       SVM_FINE_GRAIN_BUFFER = (1 << 10)
       SVM_ATOMICS = (1 << 11)
       KERNEL_READ_AND_WRITE = (1 << 12)
       MIGRATE_MEM_OBJECT_HOST = (1 << 0)
       MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED = (1 << 1)

    class ObjectType:
       OBJECT_BUFFER = 0x10F0
       OBJECT_IMAGE2D = 0x10F1
       OBJECT_IMAGE3D = 0x10F2
       OBJECT_IMAGE2D_ARRAY = 0x10F3
       OBJECT_IMAGE1D = 0x10F4
       OBJECT_IMAGE1D_ARRAY = 0x10F5
       OBJECT_IMAGE1D_BUFFER = 0x10F6
       OBJECT_PIPE = 0x10F7

    class Info:
       TYPE = 0x1100
       FLAGS = 0x1101
       SIZE = 0x1102
       HOST_PTR = 0x1103
       MAP_COUNT = 0x1104
       REFERENCE_COUNT = 0x1105
       CONTEXT = 0x1106
       ASSOCIATED_MEMOBJECT = 0x1107
       OFFSET = 0x1108
       USES_SVM_POINTER = 0x1109

    def get_size(self):
        size_c = ctypes.c_uint64(-1)
        size_size_c = ctypes.c_uint64(-1)
        call_dll(cl_lib.clGetMemObjectInfo, self.cl_mem, self.Info.SIZE, ctypes.c_uint64(8),
                 ctypes.byref(size_c), ctypes.byref(size_size_c))
        # print(f'memory object size', size_c.value)
        return size_c.value

    def copy_to(self, target):
        raise NotImplementedError

    def copy_from(self, source):
        raise NotImplementedError


class Buffer(MemObject):
    def __init__(self, device: ComputeDevice, mem_flags=MemObject.MemFlags.READ_WRITE, size=None,
                 arr: np.ndarray=None, copy_host_ptr='auto'):
        '''
        an OpenCl buffer object (e.g. global float* in kernel)

        Note: Actual memory allocation in OpenCl happens at the first access to this object

        :param cl_context:      OpenCl context
        :param mem_flags:    memory flags (CL_MEM_*)
        :param size:            size to allocate (ignored if ndarray is not None)
        :param arr:         ndarray to allocate and copy to device
        '''
        if arr is not None:
            size = arr.nbytes
            host_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
            if copy_host_ptr == 'auto' or copy_host_ptr:
                mem_flags |= self.MemFlags.COPY_HOST_PTR
        else:
            assert size is not None, 'must either give a size or a ndarray'
            host_ptr = None
            if copy_host_ptr != 'auto' and copy_host_ptr:
                raise TypeError('cannot copy host pointer: no ndarray given')
            
        cl_mem = call_dll_ret(cl_lib.clCreateBuffer, device.context_id, ctypes.c_uint64(mem_flags),
                              ctypes.c_uint64(size), host_ptr)
        super().__init__(device, cl_mem)

    def __str__(self):
        return f'{self.__class__.__name__} with size {self.size} (id {self.cl_mem})'

    def assert_ndarray_same_size(self, arr):
        assert self.size == arr.nbytes, f'buffer and array have different sizes: {self.size} {arr.nbytes}'

    def copy_to(self, target: np.ndarray, offset=0):
        '''
        copy from the OpenCl buffer to itself

        only blocking reads implemented

        :param cl_queue:    a OpenCl queue
        :param target: a ndarray to write contents to
        :param offset:      offset in cl_buffer
        :return:
        '''

        arr_pointer = target.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        size = ctypes.c_uint64(target.nbytes)
        call_dll(cl_lib.clEnqueueReadBuffer, self.device.queue_id, self.cl_mem, ctypes.c_uint64(Constants.TRUE),
                 ctypes.c_uint64(offset), size, arr_pointer,
                 ctypes.c_uint32(0), None, None)

    def copy_from(self, source: np.ndarray, offset=0):
        '''
        copy from a ndarray to itself

        only blocking writes implemented

        :param cl_queue:    a OpenCl queue
        :param source: a ndarray to read contents from
        :param offset:      offset in cl_buffer
        :return:
        '''
        # only blocking write is implemented
        arr_pointer = source.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
        size = ctypes.c_uint64(source.nbytes)
        call_dll(cl_lib.clEnqueueWriteBuffer, self.device.queue_id, self.cl_mem, ctypes.c_uint64(Constants.TRUE),
                 ctypes.c_uint64(offset), size, arr_pointer,
                 ctypes.c_uint32(0), None, None)


class Image(MemObject):
    def __init__(self, device, mem_flags=MemObject.MemFlags.READ_WRITE, format=None, shape=None,
                 arr: np.ndarray = None, is_array=False, copy_host_ptr='auto'):
        '''
        an OpenCl image object (e.g. read_only image2d_t in kernel)
        the type of image object is automatically determined from the number of dimensions of the arr or shape argument
        e.g. shape = (100, 200) => image2d_t
             arr has shape (10, 50, 30) => image3d_t
             (above with is_array=True  => image2d_array_t

        :param device:          a ComputeDevice instance
        :param mem_flags:       memory flags (OR of Image.MemFlags.*)
        :param format:          Image.Format object describing the image format (data type)
                                can be omitted for float, uint8 or uint16 data (will result in unorm data for the last two)
        :param shape:           shape of the image in numpy convention (e.g. zyx; 1, 2, or 3 dimensions), e.g. =arr.shape
        :param arr:             ndarray to allocate and copy to device
        :param is_array:        define 2d and 3d images as 1d array and 2d array (e.g. image2d_array_t in kernel)
        :param copy_host_ptr:   copy ndarray (argument arr) to device, defaults to true if ndarray given
        '''
        if arr is not None:
            assert shape is None, 'cannot give size if ndarray is given'
            shape = arr.shape
            if format is None:
                try:
                    format = self.Format(order=self.ChannelOrder.INTENSITY,
                                         type=self.default_channel_types[arr.dtype.name])
                except KeyError:
                    raise KeyError(f'no valid default channel type found for array dtype {arr.dtype.name}, possible: uint8, uint16, float32')
            host_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
            if copy_host_ptr == 'auto' or copy_host_ptr:
                mem_flags |= self.MemFlags.COPY_HOST_PTR

        else:
            if shape is None or format is None:
                raise TypeError(f'must give arguments "shape" and "format" if no ndarray is given ({shape}, {format})')
            if copy_host_ptr != 'auto' and copy_host_ptr:
                raise TypeError('cannot copy host pointer: no ndarray given')
            host_ptr = None

        ndim = len(shape)

        description = self.Description()
        if is_array and ndim == 2:
            description.image_type = self.ObjectType.OBJECT_IMAGE1D_ARRAY
        elif is_array and ndim == 3:
            description.image_type = self.ObjectType.OBJECT_IMAGE2D_ARRAY
        elif ndim == 2:
            description.image_type = self.ObjectType.OBJECT_IMAGE2D
        elif ndim == 3:
            description.image_type = self.ObjectType.OBJECT_IMAGE3D
        elif ndim == 1:
            description.image_type = self.ObjectType.OBJECT_IMAGE1D
        else:
            raise TypeError('invalid array dimensions for image')

        description.image_width = shape[-1]
        description.image_height = shape[-2] if ndim - is_array > 1 else 0
        description.image_depth = shape[-3] if ndim - is_array > 2 else 0

        if is_array:
            description.image_array_size = shape[0]

        cl_mem = call_dll_ret(cl_lib.clCreateImage, device.context_id, ctypes.c_uint64(mem_flags),
                              ctypes.byref(format), ctypes.byref(description),
                              host_ptr)
        super().__init__(device, cl_mem)
        self.shape = shape
        self.format = format
        self.ndim = len(shape)
        # self.size = np.prod(shape)*np.dtype(dtype).itemsize

    def __str__(self):
        return f'{self.__class__.__name__} with shape {self.shape} and {self.format} (id {self.cl_mem.value})'

    class ImInfo:
       IMAGE_FORMAT = 0x1110
       IMAGE_ELEMENT_SIZE = 0x1111
       IMAGE_ROW_PITCH = 0x1112
       IMAGE_SLICE_PITCH = 0x1113
       IMAGE_WIDTH = 0x1114
       IMAGE_HEIGHT = 0x1115
       IMAGE_DEPTH = 0x1116
       IMAGE_ARRAY_SIZE = 0x1117
       IMAGE_BUFFER = 0x1118
       IMAGE_NUM_MIP_LEVELS = 0x1119
       IMAGE_NUM_SAMPLES = 0x111A

    class AddressingMode:
       ADDRESS_NONE = 0x1130
       ADDRESS_CLAMP_TO_EDGE = 0x1131
       ADDRESS_CLAMP = 0x1132
       ADDRESS_REPEAT = 0x1133
       ADDRESS_MIRRORED_REPEAT = 0x1134

    class FilterMode:
       FILTER_NEAREST = 0x1140
       FILTER_LINEAR = 0x1141

    class SamplerInfo:
       SAMPLER_REFERENCE_COUNT = 0x1150
       SAMPLER_CONTEXT = 0x1151
       SAMPLER_NORMALIZED_COORDS = 0x1152
       SAMPLER_ADDRESSING_MODE = 0x1153
       SAMPLER_FILTER_MODE = 0x1154
       SAMPLER_MIP_FILTER_MODE = 0x1155
       SAMPLER_LOD_MIN = 0x1156
       SAMPLER_LOD_MAX = 0x1157

    class ChannelOrder:
       R = 0x10B0
       A = 0x10B1
       RG = 0x10B2
       RA = 0x10B3
       RGB = 0x10B4
       RGBA = 0x10B5
       BGRA = 0x10B6
       ARGB = 0x10B7
       INTENSITY = 0x10B8
       LUMINANCE = 0x10B9
       Rx = 0x10BA
       RGx = 0x10BB
       RGBx = 0x10BC
       DEPTH = 0x10BD
       DEPTH_STENCIL = 0x10BE
       sRGB = 0x10BF
       sRGBx = 0x10C0
       sRGBA = 0x10C1
       sBGRA = 0x10C2
       ABGR = 0x10C3

    class ChannelType:
       SNORM_INT8 = 0x10D0
       SNORM_INT16 = 0x10D1
       UNORM_INT8 = 0x10D2
       UNORM_INT16 = 0x10D3
       UNORM_SHORT_565 = 0x10D4
       UNORM_SHORT_555 = 0x10D5
       UNORM_INT_101010 = 0x10D6
       SIGNED_INT8 = 0x10D7
       SIGNED_INT16 = 0x10D8
       SIGNED_INT32 = 0x10D9
       UNSIGNED_INT8 = 0x10DA
       UNSIGNED_INT16 = 0x10DB
       UNSIGNED_INT32 = 0x10DC
       HALF_FLOAT = 0x10DD
       FLOAT = 0x10DE
       UNORM_INT24 = 0x10DF
       UNORM_INT_101010_2 = 0x10E0

    default_channel_types = {'uint8': ChannelType.UNORM_INT8,
                        'uint16': ChannelType.UNORM_INT16,
                        'float32': ChannelType.FLOAT}

    class Format(ctypes.Structure):
        _fields_ = [("cl_channel_order", ctypes.c_uint32),
                    ("cl_channel_type", ctypes.c_uint32)]

        def __init__(self, order, type):
            '''
            Image format object, see OpenCl documentation for allowed combinations.

            :param order:   channel order, e.g.RGBA orINTENSITY
            :param type:    channel type, e.g.SIGNED_INT16 orFLOAT
            '''
            super().__init__(cl_channel_order=order, cl_channel_type=type)

        def __str__(self):
            order_name = '[invalid]'
            for name, value in Image.ChannelOrder.__dict__.items():
                if value == self.cl_channel_order:
                    order_name = name
                    break

            type_name = '[invalid]'
            for name, value in Image.ChannelType.__dict__.items():
                if value == self.cl_channel_type:
                    type_name = name
                    break

            return f'image format ({order_name}, {type_name})'

    class Description(ctypes.Structure):
        _fields_ = [("image_type", ctypes.c_uint32),
                    ("image_width", ctypes.c_uint64),
                    ("image_height", ctypes.c_uint64),
                    ("image_depth", ctypes.c_uint64),
                    ("image_array_size", ctypes.c_uint64),
                    ("image_row_pitch", ctypes.c_uint64),
                    ("image_slice_pitch", ctypes.c_uint64),
                    ("num_mip_levels", ctypes.c_uint32),
                    ("num_samples", ctypes.c_uint32),
                    ("buffer", ctypes.c_uint64)]

        def __init__(self, **kwargs):
            super().__init__(image_row_pitch=0, image_slice_pitch=0, num_mip_levels=0, num_samples=0, buffer=0,
                             **kwargs)

    def assert_ndarray_same_type(self, ndarray):
        assert self.ndim == ndarray.ndim, f'image and array have different ndim ({self.ndim}, {ndarray.ndim})'
        assert np.allclose(self.shape,
                           ndarray.shape), f'image and array have different shapes ({self.shape}, {ndarray.shape})'

    def get_info(self, cl_image_info):
        '''
        query image information

        :param cl_image_info:   An image property, e.g. Image.ImInfo.IMAGE_WIDTH
        :return:
        '''
        info_value = ctypes.c_uint64()
        info_value_size = ctypes.c_uint64(8)
        info_size = ctypes.c_uint64()
        call_dll(cl_lib.clGetImageInfo, self.cl_mem,
                 ctypes.c_uint64(cl_image_info), info_value_size, ctypes.byref(info_value), ctypes.byref(info_size))
        return info_value.value

    def copy_to(self, target):
        '''
        copy from itself to a ndarray/Image

        notes:
        - only blocking reads implemented
        - origin and region parameters are currently not supported

        :param cl_queue:    a OpenCl queue
        :param target:      a ndarray or Image to write contents to
        :return:
        '''
        # only blocking read is implemented
        self.memory_transfer('read', target)

    def copy_from(self, source):
        '''
        copy from a ndarray/Image to itself

        notes:
        - only blocking reads implemented
        - origin and region parameters are currently not supported

        :param cl_queue:    a OpenCl queue
        :param ndarray_out: a ndarray or Image to read contents from
        :return:
        '''
        # only blocking write is implemented
        self.memory_transfer('write', source)

    def memory_transfer(self, direction, other):
        origin = (ctypes.c_uint64 * 3)()
        region = (ctypes.c_uint64 * 3)(*self.shape[::-1], *((3 - self.ndim) * (1,)))
        if issubclass(type(other), np.ndarray):
            arr_pointer = other.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
            self.assert_ndarray_same_type(other)
            if direction == 'read':
                func = cl_lib.clEnqueueReadImage
            elif direction == 'write':
                func = cl_lib.clEnqueueWriteImage
            else:
                raise TypeError('invalid transfer direction, must be "read" or "write"')

            call_dll(func, self.device.queue_id, self.cl_mem, ctypes.c_uint64(Constants.TRUE),
                     ctypes.byref(origin), ctypes.byref(region),
                     ctypes.c_uint64(0), ctypes.c_uint64(0), arr_pointer,
                     ctypes.c_uint32(0), None, None)

        elif type(other) is self.__class__:
            if direction == 'read':
                source = self
                target = other
            elif direction == 'write':
                source = other
                target = self
            else:
                raise TypeError('invalid transfer direction, must be "read" or "write"')

            call_dll(cl_lib.clEnqueueCopyImage, self.device.queue_id, source.cl_mem, target.cl_mem,
                     ctypes.byref(origin), ctypes.byref(origin), ctypes.byref(region),
                     ctypes.c_uint32(0), None, None)

        else:
            raise TypeError(f'invalid memory object of type {type(other)}, must be np.ndarray or Image')


im_format_float = Image.Format(Image.ChannelOrder.INTENSITY, Image.ChannelType.FLOAT)
im_format_uint8 = Image.Format(Image.ChannelOrder.INTENSITY, Image.ChannelType.UNORM_INT8)
im_format_uint16 = Image.Format(Image.ChannelOrder.INTENSITY, Image.ChannelType.UNORM_INT16)

def cl_valid_float_format(arr_dtype):
    if arr_dtype == np.uint8:
        return im_format_uint8
    elif arr_dtype == np.uint16:
        return im_format_uint16
    else:
        return im_format_float


def cl_cast_to_valid_float_format(image):
    # makes a copy in any case to avoid area errers from sliced arrays
    if image.dtype in (np.uint8, np.uint16, np.float32):
        return np.copy(image)
    else:
        return image.astype(np.float32)


def cl_float_format_dtype(im_dtype):
    if im_dtype in (np.uint8, np.uint16, np.float32):
        return im_dtype, cl_float_unscaler(im_dtype)
    else:
        return im_dtype, 1.


def cl_float_unscaler(im_dtype):
        if im_dtype == np.uint8:
            return 255
        elif im_dtype == np.uint16:
            return 2**16-1
        else:
            return 1
        

# ===== programs and kernels =====
class Program:
    def __init__(self, device: ComputeDevice, *sources: str, fill_data_type=None, fill_placeholders=None,
                 options='-cl-mad-enable -cl-no-signed-zeros', blocking_kernel_calls=False,):
        '''
        An OpenCl program (set of kernels) which will automatically build itself from the sources given.

        A Program is associated with a ComputeDevice and kernels can only be executed on this device.
        For using multiple queues on the same OpenCl program, multiple Program objects need to be created.

        The Kernel objects representing the compiled kernels are available as attributes with their function names
        as attribute name. See the documentation of Kernel.__call__() for their usage.

        clProgram.kernels is a dict containing all clKernel objects

        Note: All sources will be compiled into one name space.

        :param device:      a ComputeDevice (the program is actually built for its context)
        :param sources:     strings with the kernel sources (note: sources are compiled to one name space for variables)
        :param fill_data_type:  fill all occurrences of "data_type" in the sources with the OpenCl type name of the
                                numpy dtype given (see opencl.cl_numpy_dtype_str); can be used to compile dtype-specific kernels
        :param fill_placeholders:   a list with each entry a 2-tuple of (replace_string, replacement), (str() is applied to replacement)
        :param options:     build options, default: '-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros'
        :param blocking_kernel_calls: on kernel execution, always wait on the event before returning the call
        '''
        self.device = device
        self.sources = list(sources)
        self.options = bytes(options + ' -cl-kernel-arg-info', 'ascii')
        self.blocking_kernel_calls = blocking_kernel_calls

        if fill_data_type is not None:
            data_type_str = np.dtype(fill_data_type).str[-2:]
            for k in range(len(self.sources)):
                self.sources[k] = self.sources[k].replace('data_type', cl_numpy_dtype_str[data_type_str])

        if fill_placeholders is not None:
            for placeholder, replacement in fill_placeholders:
                for k in range(len(self.sources)):
                    self.sources[k] = self.sources[k].replace(placeholder, str(replacement))

        self._hash = None

        count = ctypes.c_uint32(len(sources))
        string_buffers = [ctypes.create_string_buffer(source.encode('ascii')) for source in self.sources]
        buffer_addresses = [ctypes.addressof(string_buffer) for string_buffer in string_buffers]
        strings_c = (ctypes.c_uint64 * len(buffer_addresses))(*buffer_addresses)

        self.cl_program_id = call_dll_ret(cl_lib.clCreateProgramWithSource, self.device.context_id, count, strings_c, None)

        self.kernels = {}
        self.build()
        self.init_kernels()
        if VERBOSE:
            print('created', self, sep='\n')

    def __del__(self):
        try:
            call_dll(cl_lib.clReleaseProgram, self.cl_program_id)
        except: pass

    def hash(self):
        if self._hash is None:
            _hash = hashlib.sha512()
            for source in self.sources:
                _hash.update(bytes(source, 'ascii'))
            self._hash = hashlib.hexdigest()
        return self._hash

    def __str__(self):
        text = f'===== OpenCl Program # {self.cl_program_id.value} with {len(self.kernels)} kernels ====='
        for kernel in self.kernels.values():
            text += '\n' + str(kernel)[14:]
        text += f'\n===== end kernels ====='
        return text

    class Info:
       PROGRAM_REFERENCE_COUNT = 0x1160
       PROGRAM_CONTEXT = 0x1161
       PROGRAM_NUM_DEVICES = 0x1162
       PROGRAM_DEVICES = 0x1163
       PROGRAM_SOURCE = 0x1164
       PROGRAM_BINARY_SIZES = 0x1165
       PROGRAM_BINARIES = 0x1166
       PROGRAM_NUM_KERNELS = 0x1167
       PROGRAM_KERNEL_NAMES = 0x1168
       PROGRAM_IL = 0x1169
       PROGRAM_SCOPE_GLOBAL_CTORS_PRESENT = 0x116A
       PROGRAM_SCOPE_GLOBAL_DTORS_PRESENT = 0x116B

    class BuildInfo:
       PROGRAM_BUILD_STATUS = 0x1181
       PROGRAM_BUILD_OPTIONS = 0x1182
       PROGRAM_BUILD_LOG = 0x1183
       PROGRAM_BINARY_TYPE = 0x1184
       PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE = 0x1185

    class BinaryType:
       PROGRAM_BINARY_TYPE_NONE = 0x0
       PROGRAM_BINARY_TYPE_COMPILED_OBJECT = 0x1
       PROGRAM_BINARY_TYPE_LIBRARY = 0x2
       PROGRAM_BINARY_TYPE_EXECUTABLE = 0x4

    class BuildStatus:
       BUILD_SUCCESS = 0
       BUILD_NONE = -1
       BUILD_ERROR = -2
       BUILD_IN_PROGRESS = -3

    def build(self):
        device_ids = (ctypes.c_uint64 * 1)(self.device.device_id)
        try:
            call_dll(cl_lib.clBuildProgram, self.cl_program_id, ctypes.c_uint32(1),
                     ctypes.byref(device_ids), ctypes.create_string_buffer(self.options), None, None)
        except LibError:
            build_log = ctypes.create_string_buffer(2**16)
            log_len = ctypes.c_uint64(-1)

            call_dll(cl_lib.clGetProgramBuildInfo, self.cl_program_id, self.device.device_id,
                     self.BuildInfo.PROGRAM_BUILD_LOG, ctypes.c_uint32(2**16), build_log, ctypes.byref(log_len))
            build_log = str(build_log[:log_len.value - 1], 'ascii')
            print('=== OpenCl failed while building program, build log: ===\n', build_log,
                  '\n=== end of build log ===')
            if PRINT_SOURCE_ON_BUILD_FAIL:
                print('=== Program source code: ===')
                self.print_source()
                print('=== end of source code ===')

            raise CodeError('OpenCl kernels could not be compiled due to an error in the code')

    def init_kernels(self):
        kernels = (ctypes.c_uint64 * 512)()
        num_kernels = ctypes.c_uint32()
        call_dll(cl_lib.clCreateKernelsInProgram, self.cl_program_id, ctypes.c_uint32(512),
                 ctypes.byref(kernels), ctypes.byref(num_kernels))

        self.kernel_ids = kernels[:num_kernels.value]
        for kernel in self.kernel_ids:
            cl_kernel = Kernel(self, ctypes.c_uint64(kernel))

            object.__setattr__(self, cl_kernel.function_name, cl_kernel)
            self.kernels[cl_kernel.function_name] = cl_kernel

    def __getattribute__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            if name not in ('device', 'sources', 'options', 'cl_program_id', 'kernels', 'build', 'init_kernels'):
                raise Error(f'kernel with name "{name}" not defined')
            else:
                raise

    def print_source(self):
        lineno = 1
        for source in self.sources:
            lines = source.splitlines()
            for line in lines:
                print(f'{lineno:4}:  {line}')
                lineno += 1


class Kernel:
    def __init__(self, program: Program, cl_kernel_id: ctypes.c_uint64):
        '''
        An OpenCl kernel object used in clProgram.init_kernels()
        Instances of this class are available as attributes of the corresponding Program

        :param device:          A ComputeDevice
        :param cl_program_id:   An OpenCl program id
        :param cl_kernel_id:    An OpenCl kernel id
        :param function_name:   name of the function associated with this kernel
        :param num_args:        number of arguments of this kernel
        '''
        self.program = program
        self.device = program.device
        self.cl_program_id = program.cl_program_id
        self.cl_kernel_id = cl_kernel_id
        function_name_c = ctypes.create_string_buffer(512)
        length = ctypes.c_uint64()
        call_dll(cl_lib.clGetKernelInfo, self.cl_kernel_id, Kernel.Info.KERNEL_FUNCTION_NAME,
                 ctypes.c_uint64(512), function_name_c, ctypes.byref(length))
        self.function_name = str(function_name_c[:length.value - 1], 'ascii')

        num_args_c = ctypes.c_uint32(-1)
        call_dll(cl_lib.clGetKernelInfo, self.cl_kernel_id, Kernel.Info.KERNEL_NUM_ARGS,
                 ctypes.c_uint64(4), ctypes.byref(num_args_c), ctypes.byref(length))
        self.num_args = num_args_c.value

        self.args_inspected = False
        self.args_info = []
        self.kwargs_info = {}
        self.arg_set = np.zeros(self.num_args, '?')

        self.inspect_args()

    def __del__(self):
        call_dll(cl_lib.clReleaseKernel, self.cl_kernel_id)

    def __str__(self):
        args_text = ''
        for arg_name, arg_type in self.args_info:
            args_text += arg_type + ' ' + arg_name + ', '
        return f'OpenCl kernel {self.function_name}({args_text[:-2]})'

    class Info:
       KERNEL_FUNCTION_NAME = 0x1190
       KERNEL_NUM_ARGS = 0x1191
       KERNEL_REFERENCE_COUNT = 0x1192
       KERNEL_CONTEXT = 0x1193
       KERNEL_PROGRAM = 0x1194
       KERNEL_ATTRIBUTES = 0x1195
       KERNEL_MAX_NUM_SUB_GROUPS = 0x11B9
       KERNEL_COMPILE_NUM_SUB_GROUPS = 0x11BA

    class ArgInfo:
       KERNEL_ARG_ADDRESS_QUALIFIER = 0x1196
       KERNEL_ARG_ACCESS_QUALIFIER = 0x1197
       KERNEL_ARG_TYPE_NAME = 0x1198
       KERNEL_ARG_TYPE_QUALIFIER = 0x1199
       KERNEL_ARG_NAME = 0x119A

    class ArgAddressQualifier:
       KERNEL_ARG_ADDRESS_GLOBAL = 0x119B
       KERNEL_ARG_ADDRESS_LOCAL = 0x119C
       KERNEL_ARG_ADDRESS_CONSTANT = 0x119D
       KERNEL_ARG_ADDRESS_PRIVATE = 0x119E

    class ArgAccessQualifier:
       KERNEL_ARG_ACCESS_READ_ONLY = 0x11A0
       KERNEL_ARG_ACCESS_WRITE_ONLY = 0x11A1
       KERNEL_ARG_ACCESS_READ_WRITE = 0x11A2
       KERNEL_ARG_ACCESS_NONE = 0x11A3

    class ArgTypeQualifier:
       KERNEL_ARG_TYPE_NONE = 0
       KERNEL_ARG_TYPE_CONST = (1 << 0)
       KERNEL_ARG_TYPE_RESTRICT = (1 << 1)
       KERNEL_ARG_TYPE_VOLATILE = (1 << 2)
       KERNEL_ARG_TYPE_PIPE = (1 << 3)

    class WorkGroupInfo:
       KERNEL_WORK_GROUP_SIZE = 0x11B0
       KERNEL_COMPILE_WORK_GROUP_SIZE = 0x11B1
       KERNEL_LOCAL_MEM_SIZE = 0x11B2
       KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE = 0x11B3
       KERNEL_PRIVATE_MEM_SIZE = 0x11B4
       KERNEL_GLOBAL_WORK_SIZE = 0x11B5

    class SubGroupInfo:
       KERNEL_MAX_SUB_GROUP_SIZE_FOR_NDRANGE = 0x2033
       KERNEL_SUB_GROUP_COUNT_FOR_NDRANGE = 0x2034
       KERNEL_LOCAL_SIZE_FOR_SUB_GROUP_COUNT = 0x11B8

    class ExecInfo:
       KERNEL_EXEC_INFO_SVM_PTRS = 0x11B6
       KERNEL_EXEC_INFO_SVM_FINE_GRAIN_SYSTEM = 0x11B7

    def inspect_args(self):
        for k in range(self.num_args):
            text_c = ctypes.create_string_buffer(512)
            text_length = ctypes.c_uint64(-1)

            call_dll(cl_lib.clGetKernelArgInfo, self.cl_kernel_id, ctypes.c_uint32(k),
                     self.ArgInfo.KERNEL_ARG_TYPE_NAME, ctypes.c_uint64(512), text_c, ctypes.byref(text_length))
            arg_type = str(text_c[:text_length.value - 1], 'ascii')

            call_dll(cl_lib.clGetKernelArgInfo, self.cl_kernel_id, ctypes.c_uint32(k),
                     self.ArgInfo.KERNEL_ARG_NAME, ctypes.c_uint64(512), text_c, ctypes.byref(text_length))
            arg_name = str(text_c[:text_length.value - 1], 'ascii')

            self.args_info.append((arg_name, arg_type))
            self.kwargs_info[arg_name] = k

            # print(f'kernel {self.function_name} has arg named "{arg_name}" of type {arg_type}')

    def set_args(self, *args, **kwargs):
        for k, arg in enumerate(args):
            self.set_arg(k, arg)

        num_args = len(args)
        for arg_name, arg in kwargs.items():
            try:
                k = self.kwargs_info[arg_name]
                assert k >= num_args, f'duplicate argument "{arg_name}" for function {self.function_name}()'
                # print(f'setting kwarg {arg_name} to index {k} with value {arg}')
                self.set_arg(k, arg)
            except KeyError:
                raise KeyError(f'no kwarg named "{arg_name}" possible for function {self.function_name}()')

        if not np.all(self.arg_set):
            missing = ''
            for k in range(self.num_args):
                if not self.arg_set[k]:
                    missing += f'{k}:{self.args_info[k]}, '

            if len(missing) > 0:
                raise TypeError(f'missing arguments for OpenCl kernel {self.function_name}(): {missing[:-2]}')

    def set_arg(self, k, arg):
        try:
            if issubclass(type(arg), (np.number, np.ndarray)):
                arg_value = np.ctypeslib.as_ctypes(arg)
                arg_size = arg.nbytes
            elif type(arg) in (Buffer, Image):
                arg_value = arg.cl_mem
                arg_size = 8  # nbytes of 64 bit uint
            else:
                raise TypeError(f'invalid argument type {type(arg)} for OpenCl kernel arg {k}:{self.args_info[k]}, must be in (np.number, np.ndarray, Buffer, Image)')

            try:
                if CHECK_KERNEL_ARGS:
                    self.check_arg_type(k, arg)
                call_dll(cl_lib.clSetKernelArg, self.cl_kernel_id, ctypes.c_uint32(k),
                         ctypes.c_uint64(arg_size), ctypes.byref(arg_value))
            except LibError:
                self.check_arg_type(k, arg)
                raise

            self.arg_set[k] = True
            # print(f'set arg {k}:{self.args_info[k]} to value {arg}')
        except LibError as err:
            raise Error(f'error while setting arg {k}:{self.args_info[k]} to value {arg}:\n' + str(err))

    def check_arg_type(self, k, arg):
        try:
            kernel_arg_type = self.args_info[k][1]
            if '*' in kernel_arg_type:
                if not type(arg) is Buffer:
                    raise TypeError(f'kernel buffer argument "{k}:{self.args_info[k][0]}" must be of type Buffer but is {type(arg).__name__}')

            elif 'image' in kernel_arg_type:
                if not type(arg) in (Image,):
                    raise TypeError(f'kernel image argument "{k}:{self.args_info[k][0]}" must be of type Image but is {type(arg).__name__}')

            else:
                for stripoff in (None, -1, -2):
                    try:
                        if not cl_kernel_argtypes[kernel_arg_type[:stripoff]] == arg.dtype.name:
                            raise TypeError(f'wrong dtype for argument "{k}:{self.args_info[k][0]}", is {arg.dtype}, '
                                            f'must be {cl_kernel_argtypes[kernel_arg_type.rstrip(string.digits)]}')
                        break
                    except KeyError:
                        pass

                vector_length = kernel_arg_type.lstrip(string.ascii_lowercase)
                if len(vector_length) > 0:
                    if not issubclass(type(arg), np.ndarray):
                        raise TypeError(f'kernel vector argument "{k}:{self.args_info[k][0]}" must be np.ndarray')

                    vector_length = int(vector_length)
                    if vector_length == 3:
                        vector_length = 4  # see OpenCl documentation, 3-vectors are internally 4-vectors
                    if not len(arg) == vector_length:
                        raise TypeError(f'wrong array size {len(arg)} for vector arg {kernel_arg_type}')

                else:
                    if not issubclass(type(arg), np.number):
                        raise TypeError(f'kernel number argument "{k}:{self.args_info[k][0]}" must be numpy np.number')
        except IndexError:
            raise IndexError('too many positional arguments')

    @staticmethod
    def get_global_size(global_work_size, local_work_size):
        if np.prod(local_work_size) % 16 != 0 and VERBOSE:
            print('warning: bad local size (prod not multiple of 16)', local_work_size)
        return [int(local_work_size[i] * np.ceil(global_work_size[i] / local_work_size[i])) for i in
                range(len(local_work_size))]

    def __call__(self, global_work_size, local_work_size, *args, blocking=False, global_work_offset=(), **kwargs):
        '''
        Enqueue a kernel for execution

        WARNING: the global_work_size is automatically increased to be a multiple of local_work_size (as required by OpenCl)
                 inside the kernel must be an if-clause to not execute the work items added this way
                 otherwise the kernel may crash due to invalid memory access
        example: int2 shape = get_image_dim(image);
                 int2 work_pos = (int2)(get_global_id(0), get_global_id(1));
                 if (all(work_pos < shape)){ [actual work here] }
        note:    reading invalid coordinates from images usually returns 0.0
                 writing to invalid coordinates crashes the OpenCl program (OUT_OF_RESOURCES on next use)
                 for this reason, it is safer to use get_image_dim() on the memory object which is written to

        Kernel arguments that are not supplied are used unchanged from a call before (if they were set before)

        Notes:
        - this call is not thread-safe if arguments are set
        - if a specific argument does not change in consecutive kernel calls, it can be omitted (reduces overhead)
        - kernel args that are OpenCl vectors may be passed as ndarrays (3-vectors are arrays of length 4)
          keep in mind that these have xyz order instead of the usual zyx in numpy

        :param global_work_size:    problem size in numpy axis order (e.g. zyx as from np.ndarray.shape)
        :param local_work_size:     work group size in numpy axis order (product should be a multiple of 64)
        :param blocking:            wait until kernel execution finished before returning
        :param global_work_offset:  offset for global work indices
        :param args:                positional arguments for the kernel, allowed types: Buffer, Image, numpy.ndarray, numpy.number
        :param kwargs:              keyword arguments for the kernel, allowed types: Buffer, Image, numpy.ndarray, numpy.number
        :return:                    An Event instance
        '''
        work_dim = len(global_work_size)
        global_work_size = self.get_global_size(global_work_size, local_work_size)

        global_work_offset_c = (ctypes.c_uint64 * work_dim)(*global_work_offset[::-1])
        global_work_size_c = (ctypes.c_uint64 * work_dim)(*global_work_size[::-1])
        local_work_size_c = (ctypes.c_uint64 * work_dim)(*local_work_size[::-1])

        self.set_args(*args, **kwargs)

        event = Event()
        call_dll(cl_lib.clEnqueueNDRangeKernel, self.device.queue_id, self.cl_kernel_id, ctypes.c_uint32(work_dim),
                 global_work_offset_c, global_work_size_c, local_work_size_c, None, None, ctypes.byref(event.event_id))

        if blocking or self.program.blocking_kernel_calls:
            event.wait()
        return event


class Event:
    def __init__(self):
        self.event_id = ctypes.c_uint64(0)

    def wait(self):
        assert self.event_id.value != 0, 'invalid event'
        num_events = ctypes.c_uint32(1)
        event_list = (ctypes.c_uint64 * 1)(self.event_id)
        call_dll(cl_lib.clWaitForEvents, num_events, ctypes.byref(event_list))


def load_source(python_file, *relative_path_parts):
    fname = os.path.join(os.path.dirname(python_file), *relative_path_parts)
    with open(fname) as fp:
        source = fp.read()
    #print('loaded source from', fname, '\n')#, traceback.print_stack(limit=6))
    return source