enable_language(CUDA)

include(CudaArchitectureSelector)
if(GINKGO_SPGEMM_BUILD_NSPARSE)
    add_library(spgemm-wrapper-nsparse nsparse_wrapper.cu)
    target_link_libraries(spgemm-wrapper-nsparse ginkgo)
    get_filename_component(INFERRED_CUDA_BIN_DIR "${CMAKE_CUDA_COMPILER}" DIRECTORY)
    find_path(CUDA_SAMPLE_COMMON_INCLUDE_DIR "helper_cuda.h"
        PATHS "${INFERRED_CUDA_BIN_DIR}/.."
        ENV "CUDA_ROOT"
        PATH_SUFFIXES "samples/common/inc"
        NO_DEFAULT_PATH)
    # REQUIRED parameter for find_path is not supported in old CMake versions
    if (NOT (CUDA_SAMPLE_COMMON_INCLUDE_DIR))
        message(FATAL_ERROR "Path CUDA_SAMPLE_COMMON_INCLUDE_DIR not found, it should point to CUDA_ROOT/samples/common/inc")
    endif()
    target_include_directories(spgemm-wrapper-nsparse PRIVATE nsparse/cuda-cpp/inc "${CUDA_SAMPLE_COMMON_INCLUDE_DIR}")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
        target_compile_options(spgemm-wrapper-nsparse PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    # we need double atomicAdd, i.e. Compute Capability >= 60
    cas_target_cuda_architectures(spgemm-wrapper-nsparse
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21" "30" "35" "37" "50" "52" "53")
endif()

if(GINKGO_SPGEMM_BUILD_ACSPGEMM)
    add_library(spgemm-wrapper-acspgemm ac_wrapper.cu
        ACSpGEMM/source/device/Transpose.cu
        ACSpGEMM/source/device/memory.cpp
        ACSpGEMM/source/device/Compare.cu
        ACSpGEMM/source/COO.cpp
        ACSpGEMM/source/CSR.cpp
        ACSpGEMM/source/dCSR.cpp
        ACSpGEMM/source/device/Multiply.cu)
    target_link_libraries(spgemm-wrapper-acspgemm ginkgo)
    target_include_directories(spgemm-wrapper-acspgemm PRIVATE ACSpGEMM/include ACSpGEMM/externals/include "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
        # CUDA 11 provides cub natively
        target_include_directories(spgemm-wrapper-acspgemm PRIVATE ACSpGEMM/include/external)
    endif()
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
        target_compile_options(spgemm-wrapper-acspgemm PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    # we need double atomicAdd, i.e. Compute Capability >= 60
    cas_target_cuda_architectures(spgemm-wrapper-acspgemm
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21" "30" "35" "37" "50" "52" "53")
    if (GINKGO_SPGEMM_BUILD_SPECK)
        message(FATAL_ERROR "ACSpGEMM and spECK are ABI-incompatible (ODR!) and cannot be benchmarked at the same time!")
    endif()
endif()

if (GINKGO_SPGEMM_BUILD_SPECK)
    add_library(spgemm-wrapper-speck speck_wrapper.cu
        spECK/source/GPU/Transpose.cu
        spECK/source/GPU/memory.cpp
        spECK/source/GPU/Compare.cu
        spECK/source/COO.cpp
        spECK/source/CSR.cpp
        spECK/source/dCSR.cpp
        spECK/source/GPU/Multiply.cu)
    target_link_libraries(spgemm-wrapper-speck ginkgo)
    target_include_directories(spgemm-wrapper-speck PRIVATE spECK/include spECK/externals/include "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
        # CUDA 11 provides cub natively
        target_include_directories(spgemm-wrapper-speck PRIVATE spECK/include/external)
    endif()
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
        target_compile_options(spgemm-wrapper-speck PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    # we need double atomicAdd, i.e. Compute Capability >= 60
    cas_target_cuda_architectures(spgemm-wrapper-speck
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21" "30" "35" "37" "50" "52" "53")
endif()

# To build with Kokkos, you need to set CMAKE_CXX_COMPILER=/path/to/kokkos/bin/nvcc_wrapper
if (GINKGO_SPGEMM_BUILD_KOKKOS)
    set(Kokkos_ENABLE_CUDA ON CACHE BOOL "")
    set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "")
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE "")
    set(Kokkos_CXX_STANDARD 14 CACHE ON "")
    
    add_subdirectory(kokkos)
    set(KokkosKernels_REQUIRE_DEVICES CUDA CACHE ON "")
    add_subdirectory(kokkos-kernels)
    add_library(spgemm-wrapper-kokkos kokkos_wrapper.cpp)
    target_link_libraries(spgemm-wrapper-kokkos ginkgo Kokkos::kokkoskernels)
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
        target_compile_options(spgemm-wrapper-kokkos PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()

    # we need double atomicAdd, i.e. Compute Capability >= 60
    cas_target_cuda_architectures(spgemm-wrapper-kokkos
        ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
        UNSUPPORTED "20" "21" "30" "35" "37" "50" "52" "53")
endif()

