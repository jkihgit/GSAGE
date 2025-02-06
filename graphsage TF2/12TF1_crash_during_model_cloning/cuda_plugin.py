# https://github.com/bshillingford/python-cuda-profile/blob/master/cudaprofile.py
# use --profile-from-start off when using this API

import ctypes

_cudart = ctypes.CDLL('libcudart.so')


def start():
    # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
    # the return value will unconditionally be 0. This check is just in case it changes in 
    # the future.
    ret = _cudart.cudaProfilerStart()
    if ret != 0:
        raise Exception("cudaProfilerStart() returned %d" % ret)

def stop():
    ret = _cudart.cudaProfilerStop()
    if ret != 0:
        raise Exception("cudaProfilerStop() returned %d" % ret)

def reset():    
    ret = _cudart.cudaDeviceReset()    
    if ret != 0:
        raise Exception("cudaProfilerReset() returned %d" % ret)