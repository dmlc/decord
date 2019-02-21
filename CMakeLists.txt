cmake_minimum_required(VERSION 2.8)
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

include(FindFFmpeg)
#find_package(FFmpeg COMPONENTS AVCODEC AVFORMAT AVUTIL AVDEVICE REQUIRED)

if(FFMPEG_FOUND)
    #  FFMPEG_INCLUDE_DIRS  - Include directory necessary for using the required components headers.
    #  FFMPEG_LIBRARIES     - Link these to use the required ffmpeg components.
    #  FFMPEG_DEFINITIONS   - Compiler switches required for using the required ffmpeg components.
    message("FFMPEG_INCLUDE_DIRS = ${FFMPEG_INCLUDE_DIRS} ")
    message("FFMPEG_LIBRARIES = ${FFMPEG_LIBRARIES} ")
    message("FFMPEG_DEFINITIONS = ${FFMPEG_DEFINITIONS} ")

    include_directories(${FFMPEG_INCLUDE_DIRS})

endif()

# Source file lists
file(GLOB CORE_SRCS src/decoder/*.cc src/*.cc)

file(GLOB RUNTIME_SRCS src/runtime/*.cc)


add_library(decord SHARED ${CORE_SRCS} ${RUNTIME_SRCS})

target_link_libraries(decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS} ${FFMPEG_LIBRARIES})

# Installation rules
install(TARGETS decord DESTINATION lib${LIB_SUFFIX})

# Tests
set(TEST_EXECS "")
file(GLOB TEST_SRCS tests/cpp/video/*.cc)
add_library(decord_test ${TEST_SRCS})
target_link_libraries(decord_test ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS} ${FFMPEG_LIBRARIES})