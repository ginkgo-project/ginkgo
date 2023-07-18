if (NOT CMAKE_CUDA_HOST_COMPILER AND NOT GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
elseif(GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
endif()

enable_language(CUDA)

find_package(CUDAToolkit REQUIRED)

include(cmake/Modules/CudaArchitectureSelector.cmake)

# Detect the CUDA architecture flags and propagate to all the project
cas_variable_cuda_architectures(GINKGO_CUDA_ARCH_FLAGS
    ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
    UNSUPPORTED "20" "21")

find_package(NVTX REQUIRED)

if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "The CMake CXX compiler and CUDA host compiler do not match. "
        "If you encounter any build error, especially while linking, try to use "
        "the same compiler for both.\n"
        "The CXX compiler is ${CMAKE_CXX_COMPILER} with version ${CMAKE_CXX_COMPILER_VERSION}.\n"
        "The CUDA host compiler is ${CMAKE_CUDA_HOST_COMPILER}.")
endif()

