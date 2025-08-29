cmake_minimum_required(VERSION 3.8.2)
project(decord C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindFFmpeg.cmake)
include(cmake/util/FindCUDA.cmake)
# include(cmake/util/FindCUDAArchFlags.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alternatively, use cmake -DOPTION=VALUE through command-line.
option(USE_CUDA "Build with CUDA" OFF)
option(USE_MSVC_MT "Build with MT" OFF)
option(USE_AUDIO "Build with audio support" OFF)  # Disable audio support due to FFmpeg compatibility issues

# Project
if(USE_CUDA)
project(decord C CXX CUDA)
endif(USE_CUDA)

# include directories
include_directories("include")
include_directories("3rdparty/dlpack/include")
include_directories("3rdparty/dmlc-core/include")

# Generic compilation options
if(MSVC)
# initial variables
set(DECORD_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0042 NEW)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-D_ENABLE_EXTENDED_ALIGNED_STORAGE)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  //set(CMAKE_CUDA_FLAGS "-std=c++11 ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  else()
    set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas ${CMAKE_CXX_FLAGS}")
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# 添加以下内容以支持 macOS 的特定编译选项
if(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmacosx-version-min=10.15")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mmacosx-version-min=10.15")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -mmacosx-version-min=10.15")
endif()

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "src/*.cu")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB DECORD_CORE_SRCS src/*.cc src/runtime/*.cc src/video/*.cc src/sampler/*.cc src/av_wrapper/*.cc)

# Add audio sources only if audio support is enabled
if(USE_AUDIO)
file(GLOB DECORD_AUDIO_SRCS src/audio/*.cc)
endif()

# Module rules
include(cmake/modules/FFmpeg.cmake)
include(cmake/modules/CUDA.cmake)

# Set CUDA compiler flags to support C++11
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++11")

# Targets

add_library(decord SHARED ${DECORD_CORE_SRCS} ${DECORD_FFMPEG_SRCS} ${NVDEC_SRCS} ${RUNTIME_CUDA_SRCS} ${NVDEC_CUDA_SRCS} ${DECORD_AUDIO_SRCS})

target_compile_features(decord PUBLIC cxx_std_11)

target_link_libraries(decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS})
set_property(TARGET decord PROPERTY CUDA_STANDARD 11)

# More target definitions
if(MSVC)
  target_compile_definitions(decord PRIVATE -DDECORD_EXPORTS)
endif()

# Tests
set(TEST_EXECS "")
file(GLOB_RECURSE TEST_SRCS tests/cpp/*.cc)
find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")

if(GTEST_LIB)
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    if (MSVC)
      set(GTEST_ADD_LIBS "")
    else(MSVC)
      set(GTEST_ADD_LIBS pthread dl)
    endif(MSVC)
    target_link_libraries(${__execname}
        decord ${GTEST_LIB} ${GTEST_ADD_LIBS})
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  endforeach()
  add_custom_target(cpptest DEPENDS ${TEST_EXECS})
  list(LENGTH TEST_EXECS NUM_TESTS)
  message(STATUS "Build with: " ${NUM_TESTS}  " tests")
endif()

# Installation rules
install(TARGETS decord DESTINATION lib${LIB_SUFFIX})
