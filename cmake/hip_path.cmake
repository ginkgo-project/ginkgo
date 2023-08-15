# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
        set(ENV{HIP_PATH} ${HIP_PATH})
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

find_program(GINKGO_HIPCONFIG_PATH hipconfig HINTS "${HIP_PATH}/bin")
if(GINKGO_HIPCONFIG_PATH)
    message(STATUS "Found hipconfig: ${GINKGO_HIPCONFIG_PATH}")
endif()

# We keep using NVCC/HCC for consistency with previous releases even if AMD
# updated everything to use NVIDIA/AMD in ROCM 4.1
set(GINKGO_HIP_PLATFORM_NVCC 0)
set(GINKGO_HIP_PLATFORM_HCC 0)
