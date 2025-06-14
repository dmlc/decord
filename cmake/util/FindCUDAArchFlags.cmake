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
    foreach(a IN LISTS _arch_list)
        if(a STREQUAL "native")
            list(APPEND _valid_list native)
        elseif(a MATCHES "^[0-9]+$")
            list(APPEND _valid_list ${a})
        else()
            message(FATAL_ERROR
                    "CUDA_ARCH entry '${a}' is neither a number nor 'native'.")
        endif()
    endforeach()
    set(CMAKE_CUDA_ARCHITECTURES ${_valid_list} CACHE STRING
            "GPU architectures passed to NVCC" FORCE)
endfunction()