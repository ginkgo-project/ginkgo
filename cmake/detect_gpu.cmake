################################################################################
# COPYRIGHT

# All contributions by the University of California:
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.

# All other contributions:
# Copyright (c) 2014-2017, the respective contributors
# All rights reserved.

# Caffe uses a shared copyright model: each contributor holds copyright over
# their contributions to Caffe. The project versioning records all such
# contribution and copyright details. If a contributor wants to further mark
# their specific copyright on a particular contribution, they should indicate
# their copyright solely in the commit message of the change when it is
# committed.
################################################################################
# It is from caffe: https://github.com/BVLC/caffe/blob/master/cmake/Cuda.cmake
################################################################################


# Known NVIDIA GPU achitectures Ginkgo can be compiled for.
# This list will be used for CUDA_ARCH_NAME = All option
set(ginkgo_known_gpu_archs "30;35;37;50;52;60;61")

################################################################################
# A function for automatic detection of GPUs installed
# (if autodetection is enabled)
# Usage:
#     ginkgo_detect_installed_gpus(out_variable)
function(ginkgo_detect_installed_gpus out_variable)
    set(CUDA_gpu_detect_output "" CACHE INTERNAL
        "Returned GPU architetures from ginkgo_detect_gpus tool" FORCE)
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)
    file(WRITE ${__cufile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
        "  }\n"
        "  return 0;\n"
        "}\n")

    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${__cufile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(__nvcc_res EQUAL 0)
        set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL
            "Returned GPU architetures from ginkgo_detect_gpus tool" FORCE)
    endif()

    if(NOT CUDA_gpu_detect_output)
        message(STATUS "Automatic GPU detection failed.
                       Building for all known architectures.")
        set(${out_variable} ${ginkgo_known_gpu_archs} PARENT_SCOPE)
    else()
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()


################################################################################
# A function for automatic detection of GPUs installed
# (if autodetection is enabled)
# Usage:
#     ginkgo_detect_nvcc_version(out_variable)
function(ginkgo_detect_nvcc_version out_variable)
    set(__nvfile ${PROJECT_BINARY_DIR}/detect_nvcc_version.cu)
    file(WRITE ${__nvfile} ""
        "#include <cstdio>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if (count == 0) return -1;\n"
        "  std::printf(\"%d\", __CUDACC_VER_MAJOR__);\n"
        "  return 0;\n"
        "}\n")
    execute_process(COMMAND "${CMAKE_CUDA_COMPILER}" "--run" "${__nvfile}"
                    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                    RESULT_VARIABLE __nv_res OUTPUT_VARIABLE __nv_out
                    ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(__nv_res EQUAL 0)
        set(${out_variable} ${__nv_out} PARENT_SCOPE)
    endif()
endfunction()


################################################################################
# A function for automatic detection of GPUs installed 
# (if autodetection is enabled)
# Usage:
#     check_available(arch_var,nvcc_var)
function(check_available arch_ver nvcc_ver)
    if (${arch_ver} LESS 30)
        message(FATAL_ERROR "Do not support gpu architecture < 30")
    elseif ((${arch_ver} GREATER_EQUAL 70) AND (${nvcc_ver} LESS 9))
        message(FATAL_ERROR "Volta support: Needs cuda version >= 9.0")
    endif()
endfunction()

################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#     ginkgo_select_nvcc_arch_flags(out_variable)
function(ginkgo_select_nvcc_arch_flags out_variable)
    set(__nvcc_ver 8) # default cuda 8
    ginkgo_detect_nvcc_version(__nvcc_ver)

    # List of arch names
    if (__nvcc_ver GREATER_EQUAL 9)
        set(__archs_names "Kepler" "Maxwell" "Pascal" "Volta" "All" "Manual")
        list(APPEND known_gpu_archs "70")
    else()
        set(__archs_names "Kepler" "Maxwell" "Pascal" "All" "Manual")
    endif()

    set(__archs_name_default "Auto")
    if(NOT CMAKE_CROSSCOMPILING)
        list(APPEND __archs_names "Auto")
        set(__archs_name_default "Auto")
    endif()

    # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
    set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING
        "Select target NVIDIA GPU achitecture.")
    set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )

    # verify CUDA_ARCH_NAME value
    if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
        string(REPLACE ";" ", " __archs_names "${__archs_names}")
        message(FATAL_ERROR "Only ${__archs_names} architeture names
                            are supported.")
    endif()

    if(${CUDA_ARCH_NAME} STREQUAL "Manual")
        set(CUDA_ARCH_BIN ${ginkgo_known_gpu_archs} CACHE STRING
            "Specify 'real' GPU architectures to build binaries for,
            BIN(PTX) format is supported")
        set(CUDA_ARCH_PTX "30" CACHE STRING
            "Specify 'virtual' PTX architectures
            to build PTX intermediate code for")
        mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
    else()
        unset(CUDA_ARCH_BIN CACHE)
        unset(CUDA_ARCH_PTX CACHE)
    endif()

    if(${CUDA_ARCH_NAME} STREQUAL "Kepler")
        set(__cuda_arch_bin "30 35 37")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
        set(__cuda_arch_bin "50 52")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
        set(__cuda_arch_bin "60 61")
    elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
        set(__cuda_arch_bin "70")
    elseif(${CUDA_ARCH_NAME} STREQUAL "All")
        set(__cuda_arch_bin ${ginkgo_known_gpu_archs})
    elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
        ginkgo_detect_installed_gpus(__cuda_arch_bin)
    else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
        set(__cuda_arch_bin ${CUDA_ARCH_BIN})
    endif()

    # remove dots and convert to lists
    string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
    string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
    string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
    string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
    list(REMOVE_DUPLICATES __cuda_arch_bin)
    list(REMOVE_DUPLICATES __cuda_arch_ptx)
    list(SORT __cuda_arch_bin)
    if (__cuda_arch_ptx STREQUAL "")
        set(__largest_arch "")
        list(GET __cuda_arch_bin -1 __largest_arch)
        list(APPEND __cuda_arch_ptx ${__largest_arch})
    endif()
    set(__nvcc_flags "")
    set(__nvcc_archs_readable "")

    # Tell NVCC to add binaries for the specified GPUs
    foreach(__arch ${__cuda_arch_bin})
        if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
        # User explicitly specified PTX for the concrete BIN
        check_available(${CMAKE_MATCH_1}, ${__nvcc_ver})
        check_available(${CMAKE_MATCH_2}, ${__nvcc_ver})
        list(APPEND __nvcc_flags
            -gencode=arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
        list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
        else()
        # User didn't explicitly specify PTX for the concrete BIN,
        # we assume PTX=BIN
        check_available(${__arch}, ${__nvcc_ver})
        list(APPEND __nvcc_flags
            -gencode=arch=compute_${__arch},code=sm_${__arch})
        list(APPEND __nvcc_archs_readable sm_${__arch})
        endif()
    endforeach()

    # Tell NVCC to add PTX intermediate code for the specified architectures
    foreach(__arch ${__cuda_arch_ptx})
        check_available(${__arch}, ${__nvcc_ver})
        list(APPEND __nvcc_flags
            -gencode=arch=compute_${__arch},code=compute_${__arch})
        list(APPEND __nvcc_archs_readable compute_${__arch})
    endforeach()

    string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
    set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
    set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()
