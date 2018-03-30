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
set(ginkgo_known_cuda_version "75;80;90;91")

set(ginkgo_known_gpu_archs_cuda_75 "20;21;30;32;35;37;50;52;53")
set(ginkgo_known_gpu_archs_cuda_80 "20;21;30;32;35;37;50;52;53;60;61;62")
set(ginkgo_known_gpu_archs_cuda_90 "30;32;35;37;50;52;53;60;61;62;70")
set(ginkgo_known_gpu_archs_cuda_91 "30;32;35;37;50;52;53;60;61;62;70;72")
set(ginkgo_unsupported_archs "20;21")

set(ginkgo_known_gpu_archs_name_75 "Fermi;Kepler;Maxwell")
set(ginkgo_known_gpu_archs_name_80 "Fermi;Kepler;Maxwell;Pascal")
set(ginkgo_known_gpu_archs_name_90 "Fermi;Kepler;Maxwell;Pascal;Volta")
set(ginkgo_known_gpu_archs_name_91 "Fermi;Kepler;Maxwell;Pascal;Volta")
set(ginkgo_unsupported_archs_name "Fermi")

set(cuda_arch_bin_Kepler "30;32;35;37")
set(cuda_arch_bin_Maxwell "50;52;53")
set(cuda_arch_bin_Pascal "60;61;62")
set(cuda_arch_bin_Volta_90 "70")
set(cuda_arch_bin_Volta_91 "70;72")

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
        "#include <iostream>\n"
        "int main()\n"
        "{\n"
        "  int count = 0;\n"
        "  if(cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
        "  if(count == 0) return -1;\n"
        "  for (int device = 0; device < count; ++device)\n"
        "  {\n"
        "    cudaDeviceProp prop;\n"
        "    if(cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
        "      std::cout << prop.major << prop.minor << ';';"
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
# A function for checking whether it is available
# Usage:
#     check_available(arch_var)
function(check_available arch_var)
    if(arch_var MATCHES "([0-9]+)")
        if(NOT(arch_var IN_LIST cuda_all_arch_list))
            message(FATAL_ERROR "arch ${arch_var} is not invalid")
        elseif(arch_var IN_LIST ginkgo_known_gpu_archs)
            if(arch_var IN_LIST ginkgo_unsupported_archs)
                message(FATAL_ERROR "Ginkgo does not support ${arch_var}")
            endif()
        else()
            message(FATAL_ERROR
                "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not know ${arch_var}")
        endif()
    else()
        if(arch_var IN_LIST ginkgo_known_gpu_archs_name)
            if(arch_var IN_LIST ginkgo_unsupported_archs_name)
                message(FATAL_ERROR
                    "Ginkgo does not support ${arch_var}")
            endif()
        else ()
            message(FATAL_ERROR
                "CUDA ${CMAKE_CUDA_COMPILER_VERSION} does not know ${arch_var}")
        endif()
    endif()
endfunction()


################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_OPTION
# Usage:
#     ginkgo_select_nvcc_arch_flags(out_variable)
function(ginkgo_select_nvcc_arch_flags out_variable)
    if(CMAKE_CUDA_COMPILER_VERSION MATCHES "([0-9]+).([0-9]+).(.*)")
        set(cuda_version ${CMAKE_MATCH_1}${CMAKE_MATCH_2})
    endif()
    set(ginkgo_known_gpu_archs ${ginkgo_known_gpu_archs_cuda_${cuda_version}})
    set(ginkgo_known_gpu_archs_name
        ${ginkgo_known_gpu_archs_name_${cuda_version}})
    set(cuda_arch_bin_Volta ${cuda_arch_bin_Volta_${cuda_version}})
    set(cuda_all_arch_list "")
    foreach(__ver ${ginkgo_known_cuda_version})
        list(APPEND cuda_all_arch_list ${ginkgo_known_gpu_archs_cuda_${__ver}})
    endforeach()
    list(REMOVE_DUPLICATES cuda_all_arch_list)

    # set CUDA_ARCH_OPTION strings
    set(CUDA_ARCH_OPTION "Auto" CACHE STRING
        "Select target NVIDIA GPU achitecture.
         Use ';' to seperate list.
         Arch name: Kepler, Maxwell, Pascal, Volta.
         Specified BIN, BIN(PTX), (PTX).
         MaxPTX will select the max ptx")

    set(__cuda_arch_bin "")
    set(__cuda_arch_ptx "")
    set(__bool_max_ptx "0")
    foreach(__option ${CUDA_ARCH_OPTION})
        if(__option STREQUAL "Off")
            set(${out_variable}          ""    PARENT_SCOPE)
            set(${out_variable}_readable "Off" PARENT_SCOPE)
            return()
        elseif(__option MATCHES "([0-9]+)\\(([0-9]+)\\)")
            check_available(${CMAKE_MATCH_1})
            check_available(${CMAKE_MATCH_2})
            list(APPEND __cuda_arch_bin ${__option})
        elseif(__option MATCHES "(^[0-9]+)")
            check_available(${CMAKE_MATCH_1})
            list(APPEND __cuda_arch_bin ${CMAKE_MATCH_1})
        elseif(__option MATCHES "\\(([0-9]+)\\)")
            check_available(${CMAKE_MATCH_1})
            list(APPEND __cuda_arch_ptx ${CMAKE_MATCH_1})
        elseif(__option STREQUAL "MaxPTX")
            set(__bool_max_ptx "1")
        elseif(__option STREQUAL "Auto")
            ginkgo_detect_installed_gpus(__detect_bin)
            check_available(${__detect_bin})
            list(APPEND __cuda_arch_bin ${__detect_bin})
        elseif(__option STREQUAL "All")
            list(APPEND __cuda_arch_bin ${ginkgo_known_gpu_archs})
        else()
            check_available(${__option})
            list(APPEND __cuda_arch_bin ${cuda_arch_bin_${__option}})
        endif()
    endforeach()

    list(REMOVE_DUPLICATES __cuda_arch_bin)
    set(__cuda_ptx_list "")
    set(__nvcc_flags "")
    set(__nvcc_archs_readable "")
    # Tell NVCC to add binaries for the specified GPUs
    foreach(__arch ${__cuda_arch_bin})
        if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
        # User explicitly specified PTX for the concrete BIN
            list(APPEND __nvcc_flags
                -gencode=arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
            list(APPEND __nvcc_archs_readable sm_${__arch})
            list(APPEND __cuda_ptx_list ${CMAKE_MATCH_2})
        else()
        # User didn't explicitly specify PTX for the concrete BIN,
        # we assume PTX=BIN
            list(APPEND __nvcc_flags
                -gencode=arch=compute_${__arch},code=sm_${__arch})
            list(APPEND __nvcc_archs_readable sm_${__arch})
            list(APPEND __cuda_ptx_list ${__arch})
        endif()
    endforeach()

    if(${__bool_max_ptx} OR __cuda_arch_ptx STREQUAL "")
        list(SORT __cuda_ptx_list)
        set(__largest_arch "")
        list(GET __cuda_ptx_list -1 __largest_arch)
        list(APPEND __cuda_arch_ptx ${__largest_arch})
    endif()

    foreach(__arch ${__cuda_arch_ptx})
        list(APPEND __nvcc_flags
            -gencode=arch=compute_${__arch},code=compute_${__arch})
        list(APPEND __nvcc_archs_readable compute_${__arch})
    endforeach()

    string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
    set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
    set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()
