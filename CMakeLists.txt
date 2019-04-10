cmake_minimum_required(VERSION 3.0)
project(decord C CXX)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# include directories
include_directories("include")
include_directories("3rdparty/dlpack/include")
include_directories("3rdparty/dmlc-core/include")

# initial variables
set(DECORD_LINKER_LIBS "")
set(DECORD_RUNTIME_LINKER_LIBS "")
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")
cmake_policy(SET CMP0042 NEW)

# Generic compilation options
set(CMAKE_CXX_STANDARD 11)
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
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
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")

  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
endif(MSVC)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/FindFFmpeg.cmake)
#find_package(FFmpeg COMPONENTS AVCODEC AVFORMAT AVUTIL AVDEVICE AVFILTER REQUIRED)

if(FFMPEG_FOUND)
    #  FFMPEG_INCLUDE_DIRS  - Include directory necessary for using the required components headers.
    #  FFMPEG_LIBRARIES     - Link these to use the required ffmpeg components.
    #  FFMPEG_DEFINITIONS   - Compiler switches required for using the required ffmpeg components.
    message("FFMPEG_INCLUDE_DIR = ${FFMPEG_INCLUDE_DIR} ")
    message("FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES} ")
    include_directories(${FFMPEG_INCLUDE_DIRS})
else()
    message( FATAL_ERROR "Unable to find FFMPEG automatically, please specify FFMPEG " )
endif()

# Source file lists
file(GLOB CORE_SRCS src/video/*.cc src/video/ffmpeg/*.cc src/*.cc)
message("CORE_SRCS = ${CORE_SRCS}")

file(GLOB RUNTIME_SRCS src/runtime/*.cc)


add_library(decord SHARED ${CORE_SRCS} ${RUNTIME_SRCS})

target_link_libraries(decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS} ${FFMPEG_LIBRARIES})

# Installation rules
install(TARGETS decord DESTINATION lib${LIB_SUFFIX})

# Tests
set(TEST_EXECS "")
file(GLOB TEST_SRCS tests/cpp/video/*.cc)
add_executable(test_decord ${TEST_SRCS})
target_link_libraries(test_decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS} ${FFMPEG_LIBRARIES} decord)

set(TEST_EXECS "")
file(GLOB TEST_SRCS tests/cpp/video/*.c)
add_executable(test_decord1 ${TEST_SRCS})
target_link_libraries(test_decord1 ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS} ${FFMPEG_LIBRARIES} decord)