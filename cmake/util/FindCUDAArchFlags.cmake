# This code just add -gencode arguments to CMAKE_CUDA_FLAGS based on
# the contents of CUDA_ARCH, which is a list of architectures. It can
# contain generation names from Fermi, Kepler, Maxwell, Pascal, or
# Volta, or specific architectures of the form sm_## (e.g. "sm_52")
# The default list is Maxwell, Pascal, Volta.

set(CUDA_ARCH "" CACHE STRING "List of GPU architectures to compile CUDA device code for.")

if(NOT CUDA_ARCH)
    if(${CMAKE_CUDA_FLAGS} MATCHES "--gpu-architecture|-arch[= ]|--gpu-code| -code[= ]|--generate-code|-gencode")
        message(STATUS "Using device code generation options found in CMAKE_CUDA_FLAGS")
        return()
    endif()
    set(__arch_names "Maxwell" "Pascal")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "9.0")
        list(APPEND __arch_names "Volta")
    endif()

else()
    set(__arch_names ${CUDA_ARCH})
endif()

foreach(arch ${__arch_names})
    if(${arch} STREQUAL "Fermi")
        message(FATAL_ERROR "ERROR: Fermi GPU does not have the necessary hardware video decoders")
    elseif(${arch} STREQUAL "Kepler")
        list(APPEND __arch_nums "30" "35" "37")
    elseif(${arch} STREQUAL "Maxwell")
        list(APPEND __arch_nums "50" "52")
    elseif(${arch} STREQUAL "Pascal")
        list(APPEND __arch_nums "60" "61")
    elseif(${arch} STREQUAL "Volta")
        if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
            message(FATAL_ERROR "Requested Volta architecture, but CUDA version ${CMAKE_CUDA_COMPILER_VERSION} is not new enough")
        endif()
        list(APPEND __arch_nums "70")
    elseif(${arch} MATCHES "sm_([0-9]+)")
        list(APPEND __arch_nums ${CMAKE_MATCH_1})
    else()
        message(FATAL_ERROR "ERROR: Unknown architecture ${arch} in CUDA_ARCH")
    endif()
endforeach()

if(NOT __arch_nums)
    message(FATAL_ERROR "ERROR: Don't know what GPU architectures to compile for.")
endif()

foreach(arch ${__arch_nums})
    string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()
