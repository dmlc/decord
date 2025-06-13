# SPDX-License-Identifier: Apache-2.0
# Modern drop‑in replacement for the old FindCUDA.cmake
cmake_minimum_required(VERSION 3.21)   # 3.18 introduces CUDAToolkit

include_guard(GLOBAL)

# Usage:
#   decord_find_cuda(<USE_CUDA>)
#     - <USE_CUDA>=ON           → autodetect
#     - <USE_CUDA>=/opt/cuda-12 → fixed prefix
function(decord_find_cuda USE_CUDA)
    if("${USE_CUDA}" STREQUAL "ON")
        # Rely on environment variables or the system default
        find_package(CUDAToolkit REQUIRED)
    elseif(IS_DIRECTORY "${USE_CUDA}")
        set(CUDAToolkit_ROOT "${USE_CUDA}" CACHE PATH "CUDA root dir" FORCE)
        find_package(CUDAToolkit REQUIRED)
    else()
        message(FATAL_ERROR
            "decord_find_cuda(): argument must be ON or a valid directory, got '${USE_CUDA}'")
    endif()

    # ---------------------------------------------------------------------
    # Back‑compat variables (avoid touching the cache if they already exist)
    # ---------------------------------------------------------------------
    set(_cuda_vars
        CUDA_FOUND
        CUDA_INCLUDE_DIRS
        CUDA_TOOLKIT_ROOT_DIR
        CUDA_CUDA_LIBRARY
        CUDA_CUDART_LIBRARY
        CUDA_NVRTC_LIBRARY
        CUDA_CUDNN_LIBRARY
        CUDA_CUBLAS_LIBRARY
        CUDA_NVIDIA_ML_LIBRARY
        CUDA_NVCUVID_LIBRARY
    )

    # Convenience helpers
    set(CUDA_FOUND             ${CUDAToolkit_FOUND})
    set(CUDA_INCLUDE_DIRS      ${CUDAToolkit_INCLUDE_DIRS})
    set(CUDA_TOOLKIT_ROOT_DIR  ${CUDAToolkit_ROOT})

    # Prefer imported targets where available; fall back to raw paths
    # When a library is missing we leave the variable undefined instead of
    # guessing file names – let client code do OPTIONAL_COMPONENT tests.
    set(_try_targets
            cuda_driver       CUDA_CUDA_LIBRARY
            cudart_static     CUDA_CUDART_LIBRARY
            nvrtc             CUDA_NVRTC_LIBRARY
            cublas            CUDA_CUBLAS_LIBRARY
            cudnn             CUDA_CUDNN_LIBRARY
            nvml              CUDA_NVIDIA_ML_LIBRARY
            nvcuvid           CUDA_NVCUVID_LIBRARY
    )
    foreach(pair IN LISTS _try_targets)
        list(GET pair 0 tgt)
        list(GET pair 1 var)
        if(TARGET CUDAToolkit::${tgt})
            set(${var} CUDAToolkit::${tgt})
        else()
            # Use find_library for non‑standard add‑ons (e.g. cuDNN)
            if(NOT DEFINED ${var})
                find_library(${var} NAMES ${tgt}
                        PATHS "${CUDAToolkit_LIBRARY_DIR}" "${CUDAToolkit_ROOT}"
                        PATH_SUFFIXES lib lib64 lib/stubs targets/*/lib targets/*/lib/stubs
                        NO_DEFAULT_PATH # don’t pick system‑wide stubs accidentally
                )
            endif()
        endif()
        set(${var} "${${var}}" PARENT_SCOPE)
    endforeach()

    # Propagate the rest
    foreach(v IN LISTS _cuda_vars)
        set(${v} "${${v}}" PARENT_SCOPE)
    endforeach()

    # Friendly summary (only once per configure run)
    if(NOT DEFINED _DECORD_CUDA_SUMMARY_DONE)
        set(_DECORD_CUDA_SUMMARY_DONE TRUE CACHE INTERNAL "")
        message(STATUS "CUDA toolkit root      : ${CUDA_TOOLKIT_ROOT_DIR}")
        message(STATUS "CUDA include dir       : ${CUDA_INCLUDE_DIRS}")
        foreach(v IN ITEMS CUDA_CUDART_LIBRARY CUDA_CUDA_LIBRARY
                             CUDA_NVRTC_LIBRARY CUDA_CUBLAS_LIBRARY
                             CUDA_CUDNN_LIBRARY CUDA_NVIDIA_ML_LIBRARY
                             CUDA_NVCUVID_LIBRARY)
            if(DEFINED ${v})
                message(STATUS "${v} : ${${v}}")
            endif()
        endforeach()
    endif()
endfunction()
