################################################################################
# The original version of this file was taken from Caffe
# (https://github.com/BVLC/caffe) and significantly improved to suite Ginkgo's
# needs. Caffe is distributed under the following licence:
######################### Caffe licence ########################################
# COPYRIGHT
#
# All contributions by the University of California:
# Copyright (c) 2014-2017 The Regents of the University of California (Regents)
# All rights reserved.
#
# All other contributions:
# Copyright (c) 2014-2017, the respective contributors
# All rights reserved.
#
# Caffe uses a shared copyright model: each contributor holds copyright over
# their contributions to Caffe. The project versioning records all such
# contribution and copyright details. If a contributor wants to further mark
# their specific copyright on a particular contribution, they should indicate
# their copyright solely in the commit message of the change when it is
# committed.
#
# LICENSE
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# CONTRIBUTION AGREEMENT
#
# By contributing to the BVLC/caffe repository through pull-request, comment,
# or otherwise, the contributor releases their content to the
# license and copyright terms herein.
################################################################################

# Known NVIDIA GPU achitectures Ginkgo can be compiled for.
set(ginkgo_known_cuda_version "75;80;90;91;92")

set(ginkgo_known_gpu_archs_cuda_75 "20;21;30;32;35;37;50;52;53")
set(ginkgo_known_gpu_archs_cuda_80 "20;21;30;32;35;37;50;52;53;60;61;62")
set(ginkgo_known_gpu_archs_cuda_90 "30;32;35;37;50;52;53;60;61;62;70")
set(ginkgo_known_gpu_archs_cuda_91 "30;32;35;37;50;52;53;60;61;62;70;72")
set(ginkgo_known_gpu_archs_cuda_92 "30;32;35;37;50;52;53;60;61;62;70;72")
set(ginkgo_unsupported_archs "20;21")

set(ginkgo_known_gpu_archs_name_75 "Fermi;Kepler;Maxwell")
set(ginkgo_known_gpu_archs_name_80 "Fermi;Kepler;Maxwell;Pascal")
set(ginkgo_known_gpu_archs_name_90 "Fermi;Kepler;Maxwell;Pascal;Volta")
set(ginkgo_known_gpu_archs_name_91 "Fermi;Kepler;Maxwell;Pascal;Volta")
set(ginkgo_known_gpu_archs_name_92 "Fermi;Kepler;Maxwell;Pascal;Volta")
set(ginkgo_unsupported_archs_name "Fermi")

set(cuda_arch_bin_Kepler "30;32;35;37")
set(cuda_arch_bin_Maxwell "50;52;53")
set(cuda_arch_bin_Pascal "60;61;62")
set(cuda_arch_bin_Volta_90 "70")
set(cuda_arch_bin_Volta_91 "70;72")
set(cuda_arch_bin_Volta_92 "70;72")


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
        message(STATUS "Detected GPU architectures: ${CUDA_gpu_detect_output}")
        set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
    endif()
endfunction()


################################################################################
# A function for checking whether it is available
# Usage:
#     check_available(arch_var_list valid_arch_list)
function(check_available arch_var_list valid_arch_list)
    foreach(arch_var ${arch_var_list})
        if(arch_var MATCHES "([0-9]+)")
            if(NOT(arch_var IN_LIST valid_arch_list))
                message(FATAL_ERROR "arch ${arch_var} is not invalid")
            elseif(arch_var IN_LIST ginkgo_known_gpu_archs)
                if(arch_var IN_LIST ginkgo_unsupported_archs)
                    string(CONCAT MESSAGE
                        "Ginkgo does not support GPU architecture"
                        "\"sm_${arch_var}\"")
                    message(FATAL_ERROR ${MESSAGE})
                endif()
            else()
                string(CONCAT MESSAGE
                    "\"sm_${arch_var}\" is not a valid archicecture for"
                    "NVCC ${CMAKE_CUDA_COMPILER_VERSION}")
                message(FATAL_ERROR ${MESSAGE})
            endif()
        else()
            if(arch_var IN_LIST ginkgo_known_gpu_archs_name)
                if(arch_var IN_LIST ginkgo_unsupported_archs_name)
                    message(FATAL_ERROR
                        "Ginkgo does not support ${arch_var} GPUs")
                endif()
            else ()
                string(CONCAT MESSAGE
                    "${arch_var} is not a valid GPU generation for NVCC"
                    "${CMAKE_CUDA_COMPILER_VERSION}")
                message(FATAL_ERROR ${MESSAGE})
            endif()
        endif()
    endforeach()
endfunction()


################################################################################
# Function for selecting GPU arch flags for nvcc based on architecture_list
# Usage:
#     ginkgo_select_nvcc_arch_flags(out_variable)
function(ginkgo_select_nvcc_arch_flags architecture_list out_variable)
    if(CMAKE_CUDA_COMPILER_VERSION MATCHES "([0-9]+).([0-9]+).(.*)")
        set(cuda_version ${CMAKE_MATCH_1}${CMAKE_MATCH_2})
        set(cuda_version ${cuda_version} PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Do not extract CUDA_COMPILER_VERSION:
                             ${CMAKE_CUDA_COMPILER_VERSION}")
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

    set(__cuda_arch_bin "")
    set(__cuda_arch_ptx "")
    set(__bool_max_ptx "0")
    foreach(__option ${architecture_list})
        if(__option STREQUAL "Off")
            set(${out_variable}          ""    PARENT_SCOPE)
            set(${out_variable}_readable "Off" PARENT_SCOPE)
            return()
        elseif(__option MATCHES "([0-9]+)\\(([0-9]+)\\)")
            check_available(${CMAKE_MATCH_1} "${cuda_all_arch_list}")
            check_available(${CMAKE_MATCH_2} "${cuda_all_arch_list}")
            list(APPEND __cuda_arch_bin ${__option})
        elseif(__option MATCHES "(^[0-9]+)")
            check_available(${CMAKE_MATCH_1} "${cuda_all_arch_list}")
            list(APPEND __cuda_arch_bin ${CMAKE_MATCH_1})
        elseif(__option MATCHES "\\(([0-9]+)\\)")
            check_available(${CMAKE_MATCH_1} "${cuda_all_arch_list}")
            list(APPEND __cuda_arch_ptx ${CMAKE_MATCH_1})
        elseif(__option STREQUAL "MaxPTX")
            set(__bool_max_ptx "1")
        elseif(__option STREQUAL "Auto")
            ginkgo_detect_installed_gpus(__detect_bin)
            check_available("${__detect_bin}" "${cuda_all_arch_list}")
            list(APPEND __cuda_arch_bin ${__detect_bin})
        elseif(__option STREQUAL "All")
            list(APPEND __cuda_arch_bin ${ginkgo_known_gpu_archs})
        else()
            check_available(${__option} "${cuda_all_arch_list}")
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

    string(REPLACE ";" " " __nvcc_flags "${__nvcc_flags}")
    string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
    set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
    set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()
