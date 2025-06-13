cmake_minimum_required(VERSION 3.21)          # matches helper modules
project(decord2 LANGUAGES C CXX)               # add CUDA later if requested

# ---------------------------------------------------------------------------
# User options
# ---------------------------------------------------------------------------
include(cmake/util/Util.cmake)
include(cmake/util/FindFFmpeg.cmake)
include(cmake/modules/FFmpeg.cmake)
# NOTE: do *not* reorder these two options; other code relies on them.
decord_option(USE_CUDA     "Build with CUDA" OFF)
decord_option(USE_MSVC_MT  "Build with MT runtime on MSVC" OFF)

# ---------------------------------------------------------------------------
# Optional CUDA support
# ---------------------------------------------------------------------------
if(USE_CUDA)
  # enable_language(CUDA)                     # add CUDA to this directory
  include(cmake/util/FindCUDA.cmake)
  include(cmake/util/FindCUDAArchFlags.cmake)
endif()

# ---------------------------------------------------------------------------
# FFmpeg
# ---------------------------------------------------------------------------
# find_package(FFmpeg REQUIRED MODULE COMPONENTS avcodec avformat avutil)

# ---------------------------------------------------------------------------
# Include paths *for this project only*
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
file(GLOB_RECURSE DECORD_CORE_SRCS
        src/*.cc src/runtime/*.cc src/video/*.cc
        src/sampler/*.cc src/audio/*.cc src/av_wrapper/*.cc)

if(NOT USE_CUDA)
  message(STATUS "CUDA is OFF. Filtering out CUDA-specific source files.")
  list(FILTER DECORD_CORE_SRCS EXCLUDE REGEX "/(cuda|nvcodec)/")
endif()

# Optionally add CUDA and NVDEC sources
set(DECORD_CUDA_SRCS "")
if(USE_CUDA)
  include(cmake/modules/CUDA.cmake)         # defines NVDEC_SRCS etc.
  list(APPEND DECORD_CUDA_SRCS ${NVDEC_SRCS} ${RUNTIME_CUDA_SRCS} ${NVDEC_CUDA_SRCS})
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(decord SHARED
        ${DECORD_CORE_SRCS}
        ${DECORD_CUDA_SRCS}
        ${DECORD_FFMPEG_SRCS}       # defined by cmake/modules/FFmpeg.cmake
)

target_include_directories(decord PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_include_directories(decord
        PUBLIC  $<BUILD_INTERFACE:${DECORD_PUBLIC_INCLUDES}>
)

# ---- compile features / options -------------------------------------------
target_compile_features(decord PUBLIC cxx_std_11)  # guarantees ‑std=c++11

if(MSVC)
  target_compile_definitions(decord PRIVATE
          -DWIN32_LEAN_AND_MEAN -D_CRT_SECURE_NO_WARNINGS
          -D_SCL_SECURE_NO_WARNINGS -D_ENABLE_EXTENDED_ALIGNED_STORAGE
          -DHalide_SHARED -DDECORD_EXPORTS
  )
  target_compile_options(decord PRIVATE /EHsc /MP /bigobj)
  if(USE_MSVC_MT)
    foreach(flag_var
            CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
    endforeach()
  endif()
else()
  target_compile_options(decord PRIVATE
          $<$<CONFIG:Debug>:-O0;-g>
          $<$<NOT:$<CONFIG:Debug>>:-O2;-fvisibility=hidden>
          -Wall -fPIC
          $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>
          $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,7.0>>:-faligned-new>
  )
  # Linker flags
  target_link_options(decord PRIVATE -rdynamic)
endif()

target_link_libraries(decord ${DECORD_LINKER_LIBS} ${DECORD_RUNTIME_LINKER_LIBS})
if(USE_CUDA)
  set_property(TARGET decord PROPERTY CUDA_STANDARD 11)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
file(GLOB_RECURSE TEST_SRCS tests/cpp/*.cc)
find_package(GTest QUIET)

if(GTest_FOUND AND TEST_SRCS)
  foreach(test_src IN LISTS TEST_SRCS)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name}
            PRIVATE decord GTest::gtest
            $<$<NOT:$<PLATFORM_ID:Windows>>:pthread;dl>
    )
    set_target_properties(${test_name} PROPERTIES
            EXCLUDE_FROM_ALL TRUE
            EXCLUDE_FROM_DEFAULT_BUILD TRUE
    )
  endforeach()
  add_custom_target(cpptest DEPENDS ${test_name})
  list(LENGTH TEST_SRCS NUM_TESTS)
  message(STATUS "Configured ${NUM_TESTS} C++ unit tests")
endif()

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS decord
        EXPORT decordTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

export(EXPORT decordTargets FILE decordTargets.cmake)
