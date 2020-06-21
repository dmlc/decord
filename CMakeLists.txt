cmake_minimum_required(VERSION 3.8.2)
project(decord C CXX)

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
# Alernatively, use cmake -DOPTION=VALUE through command-line.
decord_option(USE_CUDA "Build with CUDA" OFF)
decord_option(USE_MSVC_MT "Build with MT" OFF)

# Project
if(USE_CUDA)
project(decord C CXX CUDA)
endif(USE_CUDA)

# include directories
include_directories("include")
include_directories("3rdparty/dlpack/include")
include_directories("3rdparty/dmlc-core/include")

# initial variables
set(DECORD_LINKER_LIBS "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
cmake_policy(SET CMP0042 NEW)

# Generic compilation options
if(MSVC)
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
  #set(CMAKE_CUDA_FLAGS "-std=c++11 ${CMAKE_CUDA_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
  if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Build in Debug mode")
    set(CMAKE_C_FLAGS "-O0 -g -Wall -fPIC ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_C_FLAGS "-O2 -Wall -fPIC -fvisibility=hidden ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "-O0 -g -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  endif ()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "-Wno-unknown-pragmas ${CMAKE_CXX_FLAGS}")
  endif()
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "src/*.cu")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB DECORD_CORE_SRCS src/*.cc src/runtime/*.cc src/video/*.cc src/sampler/*.cc)

# Module rules
include(cmake/modules/FFmpeg.cmake)
include(cmake/modules/CUDA.cmake)

# Targets

add_library(decord SHARED ${DECORD_CORE_SRCS} ${DECORD_FFMPEG_SRCS} ${NVDEC_SRCS} ${RUNTIME_CUDA_SRCS} ${NVDEC_CUDA_SRCS})

# target_compile_features(decord PUBLIC cxx_std_11)

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
