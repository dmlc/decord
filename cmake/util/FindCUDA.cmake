# Modernized FindCUDA.cmake using CUDAToolkit
#
# This script is a drop-in replacement for the deprecated find_cuda macro.
# It uses the modern find_package(CUDAToolkit) command, which is available
# in CMake 3.18 and later.
#
# Usage:
#   find_cuda(<USE_CUDA>)
#     - <USE_CUDA>=ON           -> Autodetect CUDA from system path/environment.
#     - <USE_CUDA>=/path/to/cuda -> Use a specific CUDA installation.
#
# Provides the same legacy variables as the old script for backward compatibility.
#
cmake_minimum_required(VERSION 3.18)
include_guard(GLOBAL)

function(find_cuda USE_CUDA_HINT)
  # --- 1. Find the CUDAToolkit package ---
  # We list all desired components. `find_package` will find them if they exist.
  if("${USE_CUDA_HINT}" STREQUAL "ON")
    # Standard search
    find_package(CUDAToolkit QUIET COMPONENTS cudart nvrtc cublas cudnn nvml nvcuvid)
  elseif(IS_DIRECTORY "${USE_CUDA_HINT}")
    # Hinted search
    find_package(CUDAToolkit QUIET HINTS "${USE_CUDA_HINT}" COMPONENTS cudart nvrtc cublas cudnn nvml nvcuvid)
  else()
    message(FATAL_ERROR "find_cuda(): Argument must be ON or a valid directory, got '${USE_CUDA_HINT}'")
  endif()

  if(NOT CUDAToolkit_FOUND)
    message(STATUS "CUDA not found.")
    set(CUDA_FOUND FALSE PARENT_SCOPE)
    return()
  endif()

  # --- 2. Populate Legacy Variables for Backward Compatibility ---
  # This section makes the script a drop-in replacement by creating the old
  # variables that the rest of the project expects.

  # Primary variables
  set(CUDA_FOUND TRUE PARENT_SCOPE)
  set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS} PARENT_SCOPE)
  set(CUDA_TOOLKIT_ROOT_DIR ${CUDAToolkit_ROOT} PARENT_SCOPE)

  # A map of CUDAToolkit components to the legacy variable names
  set(_component_map
          cuda_driver   CUDA_CUDA_LIBRARY
          cudart        CUDA_CUDART_LIBRARY
          nvrtc         CUDA_NVRTC_LIBRARY
          cudnn         CUDA_CUDNN_LIBRARY
          cublas        CUDA_CUBLAS_LIBRARY
          nvml          CUDA_NVIDIA_ML_LIBRARY
          nvcuvid       CUDA_NVCUVID_LIBRARY
  )

  # Loop through the map and set the legacy variables if the target exists
  foreach(pair IN LISTS _component_map)
    list(GET pair 0 component_name)
    list(GET pair 1 legacy_variable_name)

    if(TARGET CUDAToolkit::${component_name})
      # The modern way is to use the imported target directly
      set(${legacy_variable_name} "CUDAToolkit::${component_name}" PARENT_SCOPE)
    else()
      # If the component target doesn't exist, leave the legacy variable undefined
      # to match the old behavior.
      set(${legacy_variable_name} "" PARENT_SCOPE)
    endif()
  endforeach()

  # --- 3. Print a summary (optional but helpful) ---
  message(STATUS "Modernized FindCUDA: Found CUDA ${CUDAToolkit_VERSION}")
  message(STATUS "Found CUDA_TOOLKIT_ROOT_DIR= ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "Found CUDA_INCLUDE_DIRS= ${CUDA_INCLUDE_DIRS}")
  message(STATUS "Found CUDA_CUDART_LIBRARY= ${CUDA_CUDART_LIBRARY}")
  message(STATUS "Found CUDA_NVRTC_LIBRARY= ${CUDA_NVRTC_LIBRARY}")
  message(STATUS "Found CUDA_CUBLAS_LIBRARY= ${CUDA_CUBLAS_LIBRARY}")
  message(STATUS "Found CUDA_CUDNN_LIBRARY= ${CUDA_CUDNN_LIBRARY}")
  message(STATUS "Found CUDA_NVCUVID_LIBRARY= ${CUDA_NVCUVID_LIBRARY}")
  message(STATUS "Found CUDA_NVIDIA_ML_LIBRARY= ${CUDA_NVIDIA_ML_LIBRARY}")

endfunction()