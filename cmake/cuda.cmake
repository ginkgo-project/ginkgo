enable_language(CUDA)

if(MSVC)
    # MSVC can not find CUDA automatically
    # Use CUDA_COMPILER PATH to define the CUDA TOOLKIT ROOT DIR
    string(REPLACE "/bin/nvcc.exe" "" CMAKE_CUDA_ROOT_DIR ${CMAKE_CUDA_COMPILER})
    if("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" STREQUAL "")
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/include")
    endif()
    if("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" STREQUAL "")
        set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/lib/x64")
    endif()
endif()

include(cmake/Modules/CudaArchitectureSelector.cmake)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Detect the CUDA architecture flags and propagate to all the project
cas_variable_cuda_architectures(GINKGO_CUDA_ARCH_FLAGS
    ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
    UNSUPPORTED "20" "21")

if (CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
    find_package(NVHPC REQUIRED
        HINTS
        $ENV{NVIDIA_PATH}
        ${CMAKE_CUDA_COMPILER}/../../..
        )

    set(CUDA_RUNTIME_LIBS_DYNAMIC ${NVHPC_CUDART_LIBRARY})
    set(CUDA_RUNTIME_LIBS_STATIC ${NVHPC_CUDART_LIBRARY_STATIC})
    set(CUBLAS ${NVHPC_CUBLAS_LIBRARY})
    set(CUSPARSE ${NVHPC_CUSPARSE_LIBRARY})
    set(CURAND ${NVHPC_CURAND_LIBRARY})
    set(CUFFT ${NVHPC_CUFFT_LIBRARY})
else()
    find_library(CUDA_RUNTIME_LIBS_DYNAMIC cudart
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUDA_RUNTIME_LIBS_STATIC cudart_static
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

    # CUDA 10.1/10.2 put cublas, cublasLt, cudnn in /usr/lib/<arch>-linux-gnu/, but
    # others (<= 10.0 or >= 11) put them in cuda own directory
    # If the environment installs several cuda including 10.1/10.2, cmake will find
    # the 10.1/10.2 .so files when searching others cuda in the default path.
    # CMake already puts /usr/lib/<arch>-linux-gnu/ after cuda own directory in the
    # `CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES`, so we always put NO_DEFAULT_PATH here.
    find_library(CUBLAS cublas
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)
    find_library(CUSPARSE cusparse
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CURAND curand
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(CUFFT cufft
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    find_library(NVTX nvToolsExt
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

# MSVC nvcc uses static cudartlibrary by default, and other platforms use shared cudartlibrary.
# add `-cudart shared` or `-cudart=shared` according system into CMAKE_CUDA_FLAGS
# to force nvcc to use dynamic cudart library in MSVC.
if(MSVC)
    if("${CMAKE_CUDA_FLAGS}" MATCHES "-cudart(=| )shared")
        set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_DYNAMIC}" CACHE STRING "Path to a library" FORCE)
    else()
        set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_STATIC}" CACHE STRING "Path to a library" FORCE)
    endif()
else()
    set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_DYNAMIC}" CACHE STRING "Path to a library" FORCE)
endif()

if (NOT CMAKE_CUDA_HOST_COMPILER AND NOT GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
elseif(GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
endif()

if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "The CMake CXX compiler and CUDA host compiler do not match. "
        "If you encounter any build error, especially while linking, try to use "
        "the same compiler for both.\n"
        "The CXX compiler is ${CMAKE_CXX_COMPILER} with version ${CMAKE_CXX_COMPILER_VERSION}.\n"
        "The CUDA host compiler is ${CMAKE_CUDA_HOST_COMPILER}.")
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION
    MATCHES "9.2" AND CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*" )
    ginkgo_extract_clang_version(${CMAKE_CUDA_HOST_COMPILER} GINKGO_CUDA_HOST_CLANG_VERSION)

    if (GINKGO_CUDA_HOST_CLANG_VERSION MATCHES "5\.0.*")
        message(FATAL_ERROR "There is a bug between nvcc 9.2 and clang 5.0 which create a compiling issue."
            "Consider using a different CUDA host compiler or CUDA version.")
    endif()
endif()
