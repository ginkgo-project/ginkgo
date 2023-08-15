# SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
#
# SPDX-License-Identifier: BSD-3-Clause

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

include(cmake/Modules/CudaArchitectureSelector.cmake)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    # Detect the CUDA architecture and propagate it to the entire project
    cas_variable_cmake_cuda_architectures(CMAKE_CUDA_ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES})
endif()

find_package(NVTX REQUIRED)

if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "The CMake CXX compiler and CUDA host compiler do not match. "
        "If you encounter any build error, especially while linking, try to use "
        "the same compiler for both.\n"
        "The CXX compiler is ${CMAKE_CXX_COMPILER} with version ${CMAKE_CXX_COMPILER_VERSION}.\n"
        "The CUDA host compiler is ${CMAKE_CUDA_HOST_COMPILER}.")
endif()
