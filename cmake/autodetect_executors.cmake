# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

set(GINKGO_HAS_OMP OFF)
set(GINKGO_HAS_MPI OFF)
set(GINKGO_HAS_CUDA OFF)
set(GINKGO_HAS_DPCPP OFF)
set(GINKGO_HAS_HIP OFF)

include(CheckLanguage)

if (NOT DEFINED GINKGO_BUILD_OMP)
    find_package(OpenMP 3.0)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Enabling OpenMP executor")
        set(GINKGO_HAS_OMP ON)
    endif()
endif()

if (NOT DEFINED GINKGO_BUILD_MPI)
    find_package(MPI 3.1 COMPONENTS CXX)
    if(MPI_FOUND)
        message(STATUS "Enabling MPI support")
        set(GINKGO_HAS_MPI ON)
    endif()
endif()

if (NOT DEFINED GINKGO_BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "Enabling CUDA executor")
        set(GINKGO_HAS_CUDA ON)
    endif()
endif()

if (NOT DEFINED GINKGO_BUILD_HIP)
    if(GINKGO_HIPCONFIG_PATH)
        message(STATUS "Enabling HIP executor")
        set(GINKGO_HAS_HIP ON)
    endif()
endif()

if (NOT DEFINED GINKGO_BUILD_DPCPP)
    try_compile(GKO_CAN_COMPILE_DPCPP ${PROJECT_BINARY_DIR}/dpcpp
        SOURCES ${PROJECT_SOURCE_DIR}/dpcpp/test_dpcpp.dp.cpp
        CXX_STANDARD 17)
    if (GKO_CAN_COMPILE_DPCPP)
        message(STATUS "Enabling DPCPP executor")
        set(GINKGO_HAS_DPCPP ON)
    endif()
endif()
