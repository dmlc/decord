cmake_minimum_required(VERSION 3.21)
project(decord2 LANGUAGES C CXX)

# ---------------------------------------------------------------------------
# User options & Core Dependencies
# ---------------------------------------------------------------------------
include(cmake/util/Util.cmake)
# Find FFmpeg (required for all builds, regardless of CUDA)
include(cmake/util/FindFFmpeg.cmake)
include(cmake/modules/FFmpeg.cmake)
include(cmake/modules/VideoToolbox.cmake)

decord_option(USE_CUDA "Build with CUDA" OFF)
decord_option(USE_MSVC_MT "Build with MT runtime on MSVC" OFF)
decord_option(USE_VIDEOTOOLBOX "Build with VideoToolbox support (macOS only)" OFF)

# ---------------------------------------------------------------------------
# Source File Discovery
# ---------------------------------------------------------------------------

# Gather all non-CUDA C++ source files first.
file(GLOB_RECURSE DECORD_CORE_SRCS
        "src/runtime/*.cc"
        "src/video/ffmpeg/*.cc"
        "src/video/logging.cc"
        "src/video/storage_pool.cc"
        "src/video/video_interface.cc"
        "src/video/video_loader.cc"
        "src/video/video_reader.cc"
        "src/sampler/*.cc"
        "src/audio/*.cc"
        "src/av_wrapper/*.cc"
        "src/segmenter/*.cc"
)

# Filter out any remaining cuda files just in case (robustness)
if(NOT USE_CUDA)
  list(FILTER DECORD_CORE_SRCS EXCLUDE REGEX "/runtime/cuda/")
endif()

# The VideoToolbox device API is macOS-only (it pulls in CoreFoundation and
# uses C11 aligned_alloc, which MSVC does not provide). On Apple it is wired
# up explicitly by VideoToolbox.cmake; keep its stub out of the core glob on
# every other platform so Windows/Linux builds don't try to compile it.
if(NOT APPLE)
  list(FILTER DECORD_CORE_SRCS EXCLUDE REGEX "videotoolbox_device_api")
endif()


# ---------------------------------------------------------------------------
# Library Target Definition
# ---------------------------------------------------------------------------
add_library(decord SHARED ${DECORD_CORE_SRCS} ${VIDEOTOOLBOX_SRCS})

# --- Modern Target-Based Properties ---

# Set the include directories needed to compile the library
target_include_directories(decord
        PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dlpack/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/dmlc-core/include"
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# Link against FFmpeg (always required)
if(FFMPEG_LIBRARIES)
  target_link_libraries(decord
          PRIVATE
          ${FFMPEG_LIBRARIES}
  )
endif()

# Link against macOS frameworks and any extra libs collected by modules
if (DECORD_LINKER_LIBS)
  target_link_libraries(decord PRIVATE ${DECORD_LINKER_LIBS})
endif()

# Set the C++ standard and compiler options
target_compile_features(decord PUBLIC cxx_std_11)

if(MSVC)
  target_compile_definitions(decord PRIVATE -DDECORD_EXPORTS)
  # ... other MSVC options ...
else()
  target_compile_options(decord PRIVATE -fvisibility=hidden -Wall -fPIC)
  target_compile_options(decord PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>)
  target_compile_options(decord PRIVATE $<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_GREATER:$<CXX_COMPILER_VERSION>,7.0>>:-faligned-new>)
  target_link_options(decord PRIVATE -rdynamic)
endif()

# ---------------------------------------------------------------------------
# Optional CUDA Logic (The Modern Way)
# ---------------------------------------------------------------------------
if(USE_CUDA)
  message(STATUS "USE_CUDA is ON. Configuring CUDA support.")
  # 1. Enable the CUDA language. This finds the compiler automatically.
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # 2. Add CUDA-specific source files to our target
  file(GLOB_RECURSE CUDA_RUNTIME_SRCS "src/runtime/cuda/*.cc")
  file(GLOB_RECURSE CUDA_NVDEC_SRCS "src/video/nvcodec/*.cc")
  file(GLOB_RECURSE CUDA_NVDEC_CU_SRCS "src/improc/*.cu")
  target_sources(decord PRIVATE ${CUDA_RUNTIME_SRCS} ${CUDA_NVDEC_SRCS} ${CUDA_NVDEC_CU_SRCS})

  # 3. Add the CUDA compile definition
  target_compile_definitions(decord PRIVATE DECORD_USE_CUDA)

  # 4. Link against the modern CUDAToolkit imported targets.
  target_link_libraries(decord PRIVATE
          CUDA::cudart
          CUDA::nvrtc
          CUDA::cublas
          CUDA::nvml
  )

  # CUDA Driver API (cuInit / cuCtx* / cuModule* / cuLaunchKernel / cuMemset*).
  # MUST be linked, otherwise libdecord.so/decord.dll ends up with undefined
  # cu* symbols and `import decord` fails (e.g. "undefined symbol:
  # cuModuleLoadData"). We link the toolkit's driver STUB, which records the
  # right SONAME in DT_NEEDED (libcuda.so.1 on Linux, imports nvcuda.dll via
  # cuda.lib on Windows) WITHOUT bundling the real driver. auditwheel/delvewheel
  # exclude libcuda/nvcuda so the host driver provides it at runtime.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(decord PRIVATE CUDA::cuda_driver)
  endif()

  # NVDEC (cuvid*) lives in libnvcuvid.so.1 / nvcuvid.dll, shipped with the
  # DRIVER (not the toolkit). Link it so DT_NEEDED carries libnvcuvid.so.1 /
  # nvcuvid.dll; auditwheel/delvewheel exclude it, so it resolves from the
  # host driver at runtime. The runtime preloader in decord/_ffi/base.py is a
  # belt-and-suspenders fallback when the lib is absent at build time.
  if(WIN32)
    # On Windows nvcuvid.lib is not in the toolkit; the workflow synthesizes an
    # import library from scripts/nvcuvid.def and passes -DCUDA_NVCUVID_LIBRARY.
    find_library(CUDA_NVCUVID_LIBRARY nvcuvid
            HINTS
            ${CUDAToolkit_LIBRARY_DIR}
            ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
            ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32)
    if(CUDA_NVCUVID_LIBRARY)
      message(STATUS "Found nvcuvid: ${CUDA_NVCUVID_LIBRARY}")
      target_link_libraries(decord PRIVATE ${CUDA_NVCUVID_LIBRARY})
    else()
      message(FATAL_ERROR "nvcuvid.lib not found; it is required to link NVDEC on Windows")
    endif()
  elseif(NOT APPLE)
    # Link by exact soname so DT_NEEDED records libnvcuvid.so.1 without needing
    # a (non-existent) toolkit dev symlink. The driver lib is present in the
    # manylinux build container (installed in CIBW_BEFORE_ALL_LINUX).
    file(GLOB _DECORD_NVCUVID_SO
            /usr/lib64/libnvcuvid.so.1
            /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1
            /usr/lib/aarch64-linux-gnu/libnvcuvid.so.1
            /usr/local/cuda/lib64/libnvcuvid.so.1)
    if(_DECORD_NVCUVID_SO)
      message(STATUS "Linking NVDEC via -l:libnvcuvid.so.1")
      target_link_libraries(decord PRIVATE "-l:libnvcuvid.so.1")
    else()
      message(STATUS "libnvcuvid.so.1 not found at build time; NVDEC will be resolved at runtime by the decord/_ffi/base.py preloader")
    endif()
  endif()



  # 5. Set the CUDA standard
  set_property(TARGET decord PROPERTY CUDA_STANDARD 11)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
find_package(GTest QUIET)
if(GTest_FOUND)
  enable_testing()
  file(GLOB_RECURSE TEST_SRCS "tests/cpp/*.cc")
  foreach(test_src IN LISTS TEST_SRCS)
    get_filename_component(test_name "${test_src}" NAME_WLE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE decord GTest::gtest)
    if(UNIX AND NOT APPLE AND NOT MSVC)
      target_link_libraries(${test_name} PRIVATE pthread dl)
    elseif(APPLE)
      target_link_libraries(${test_name} PRIVATE pthread)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
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

install(DIRECTORY include/decord DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
export(EXPORT decordTargets FILE "${CMAKE_CURRENT_BINARY_DIR}/decord-config.cmake")