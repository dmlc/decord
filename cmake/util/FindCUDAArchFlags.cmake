# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.21)
include_guard(GLOBAL)

# ---- PUBLIC CACHE VARIABLE (unchanged) ---------------------------------
set(CUDA_ARCH "" CACHE STRING
    "Comma/space separated list of GPU architectures (e.g. 52;60,61;80). \
     Empty = sensible default for the detected CUDA version.")

# ---- IMPLEMENTATION ----------------------------------------------------
function(decord_set_cuda_architectures)
    find_package(CUDAToolkit QUIET)
    if(CUDA_ARCH)
        message(STATUS "Using user-specified CUDA_ARCH=${CUDA_ARCH}")
        string(REPLACE "," ";" _arch_list "${CUDA_ARCH}")
    else()
        set(_arch_list ${CUDAToolkit_DEFAULT_ARCHITECTURES})
        message(STATUS "Auto-detected CUDA architectures: ${_arch_list}")
    endif()
    if(NOT _arch_list)
        message(FATAL_ERROR "Could not auto-detect GPU architecture and CUDA_ARCH was not set. "
                "Please ensure your NVIDIA driver and CUDA toolkit are installed correctly, "
                "or manually set the architecture with -DCUDA_ARCH=XX.")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES ${_arch_list} CACHE STRING
            "GPU architectures passed to NVCC" FORCE)
endfunction()