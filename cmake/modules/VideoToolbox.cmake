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

# VideoToolbox Module for macOS GPU acceleration
if(APPLE)
  message(STATUS "Build with VideoToolbox support for macOS GPU acceleration")

  # Find VideoToolbox and CoreVideo frameworks
  find_library(VIDEOTOOLBOX_LIBRARY VideoToolbox)
  find_library(COREVIDEO_LIBRARY CoreVideo)
  find_library(COREFOUNDATION_LIBRARY CoreFoundation)
  find_library(COREMEDIA_LIBRARY CoreMedia)
  find_library(METAL_LIBRARY Metal)

  if(VIDEOTOOLBOX_LIBRARY AND COREVIDEO_LIBRARY AND COREFOUNDATION_LIBRARY AND COREMEDIA_LIBRARY AND METAL_LIBRARY)
    message(STATUS "Found VideoToolbox: ${VIDEOTOOLBOX_LIBRARY}")
    message(STATUS "Found CoreVideo: ${COREVIDEO_LIBRARY}")
    message(STATUS "Found CoreFoundation: ${COREFOUNDATION_LIBRARY}")
    message(STATUS "Found CoreMedia: ${COREMEDIA_LIBRARY}")
    message(STATUS "Found Metal: ${METAL_LIBRARY}")

    # Add VideoToolbox source files (use absolute paths to avoid scope/path issues)
    set(_DECORD_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(GLOB _VTB_DECODER_SRCS "${_DECORD_ROOT_DIR}/src/video/videotoolbox/*.cc")
    set(VIDEOTOOLBOX_SRCS ${_VTB_DECODER_SRCS})
    list(APPEND VIDEOTOOLBOX_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime/videotoolbox_device_api.cc")

    # Add definitions
    add_definitions(-DDECORD_USE_VIDEOTOOLBOX)

    # Add libraries to a variable that the main target will link against
    list(APPEND DECORD_LINKER_LIBS ${VIDEOTOOLBOX_LIBRARY})
    list(APPEND DECORD_LINKER_LIBS ${COREVIDEO_LIBRARY})
    list(APPEND DECORD_LINKER_LIBS ${COREFOUNDATION_LIBRARY})
    list(APPEND DECORD_LINKER_LIBS ${COREMEDIA_LIBRARY})
    list(APPEND DECORD_LINKER_LIBS ${METAL_LIBRARY})

    # Mark that VideoToolbox was configured successfully
    set(VIDEOTOOLBOX_FOUND TRUE PARENT_SCOPE)

    set(VIDEOTOOLBOX_FOUND TRUE)
  else()
    message(WARNING "VideoToolbox libraries not found. GPU acceleration will not be available.")
    set(VIDEOTOOLBOX_FOUND FALSE)
  endif()
else()
  message(STATUS "VideoToolbox not available on this platform")
  set(VIDEOTOOLBOX_FOUND FALSE)
endif()