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

# CUDA Module
find_cuda(${USE_CUDA})

if(CUDA_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(${CUDA_INCLUDE_DIRS})
  add_definitions(-DDECORD_USE_CUDA)
endif(CUDA_FOUND)

if(USE_CUDA)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA, USE_CUDA=" ${USE_CUDA})
  endif()
  if(NOT CUDA_NVCUVID_LIBRARY)
    message(FATAL_ERROR "Cannot find libnvcuvid, you may need to manually register and download at https://developer.nvidia.com/nvidia-video-codec-sdk. Then copy libnvcuvid to cuda_toolkit_root/lib64/" )
  endif()
  message(STATUS "Build with CUDA support")
  file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
  file(GLOB NVDEC_SRCS src/video/nvcodec/*.cc)
  file(GLOB NVDEC_CUDA_SRCS src/improc/*.cu)

  list(APPEND DECORD_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_NVRTC_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_NVIDIA_ML_LIBRARY})
  list(APPEND DECORD_RUNTIME_LINKER_LIBS ${CUDA_NVCUVID_LIBRARY})

else(USE_CUDA)
  message(STATUS "CUDA disabled, no nvdec capabilities will be enabled...")
  set(NVDEC_SRCS "")
  set(RUNTIME_CUDA_SRCS "")
endif(USE_CUDA)
