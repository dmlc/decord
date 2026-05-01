# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# Modern CUDA module for Decord (CMake >= 3.10)

# =========================
# CUDA ENABLE
# =========================
if(USE_CUDA)

  enable_language(CUDA)

  # Usa il nuovo sistema CMake
  find_package(CUDAToolkit REQUIRED)

  set(CUDA_FOUND TRUE)

  # Include dirs
  set(CUDA_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
  include_directories(${CUDA_INCLUDE_DIRS})

  # Definizione macro
  add_definitions(-DDECORD_USE_CUDA)

  # =========================
  # LIBRERIE CUDA
  # =========================

  # Librerie moderne (target CMake)
  set(CUDA_CUDART_LIBRARY CUDA::cudart)
  set(CUDA_CUDA_LIBRARY CUDA::cuda_driver)
  set(CUDA_NVRTC_LIBRARY CUDA::nvrtc)

  # NVML (non sempre presente come target)
  find_library(
    CUDA_NVIDIA_ML_LIBRARY
    NAMES nvidia-ml
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/lib64
      /usr/local/cuda/targets/x86_64-linux/lib/stubs
  )

  # NVDEC (fondamentale per Decord)
  find_library(
    CUDA_NVCUVID_LIBRARY
    NAMES nvcuvid
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/lib64
      /usr/local/cuda/targets/x86_64-linux/lib
  )

  if(NOT CUDA_NVCUVID_LIBRARY)
    message(FATAL_ERROR
      "Cannot find libnvcuvid. Installa il Video Codec SDK oppure verifica:\n"
      "/usr/lib/x86_64-linux-gnu/libnvcuvid.so"
    )
  endif()

  message(STATUS "Build with CUDA support")

  # =========================
  # SORGENTI CUDA
  # =========================

  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  file(GLOB NVDEC_SRCS src/video/nvcodec/*.cc)
  file(GLOB NVDEC_CUDA_SRCS src/improc/*.cu)

  # =========================
  # LINKER
  # =========================

  list(APPEND DECORD_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})

  list(APPEND DECORD_RUNTIME_LINKER_LIBS
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_NVRTC_LIBRARY}
    ${CUDA_NVIDIA_ML_LIBRARY}
    ${CUDA_NVCUVID_LIBRARY}
  )

else()

  message(STATUS "CUDA disabled, no nvdec capabilities will be enabled...")

  set(CUDA_FOUND FALSE)
  set(NVDEC_SRCS "")
  set(RUNTIME_CUDA_SRCS "")
  set(NVDEC_CUDA_SRCS "")

endif()