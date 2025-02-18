cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
add_library(ginkgo_cuda $<TARGET_OBJECTS:ginkgo_cuda_device> "")
include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/csr_kernels.instantiate.cpp CSR_INSTANTIATE)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip matrix/fbcsr_kernels.instantiate.cpp FBCSR_INSTANTIATE)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip solver/batch_bicgstab_launch.instantiate.cpp BATCH_BICGSTAB_INSTANTIATE1)
add_instantiation_files(. solver/batch_bicgstab_launch.instantiate.cu BATCH_BICGSTAB_INSTANTIATE2)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/cuda_hip solver/batch_cg_launch.instantiate.cpp BATCH_CG_INSTANTIATE1)
add_instantiation_files(. solver/batch_cg_launch.instantiate.cu BATCH_CG_INSTANTIATE2)
# we don't split up the dense kernels into distinct compilations
list(
    APPEND
    GKO_UNIFIED_COMMON_SOURCES
    ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp
)
target_sources(
    ginkgo_cuda
    PRIVATE
        base/device.cpp
        base/exception.cpp
        base/executor.cpp
        base/memory.cpp
        base/nvtx.cpp
        base/scoped_device_id.cpp
        base/stream.cpp
        base/timer.cpp
        base/version.cpp
        ${CSR_INSTANTIATE}
        ${FBCSR_INSTANTIATE}
        matrix/fft_kernels.cu
        preconditioner/batch_jacobi_kernels.cu
        solver/batch_bicgstab_kernels.cu
        ${BATCH_BICGSTAB_INSTANTIATE1}
        ${BATCH_BICGSTAB_INSTANTIATE2}
        solver/batch_cg_kernels.cu
        ${BATCH_CG_INSTANTIATE1}
        ${BATCH_CG_INSTANTIATE2}
        solver/lower_trs_kernels.cu
        solver/upper_trs_kernels.cu
        ${GKO_UNIFIED_COMMON_SOURCES}
        ${GKO_CUDA_HIP_COMMON_SOURCES}
)
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_CUDA_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 32)
        list(APPEND GKO_CUDA_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_CUDA_JACOBI_BLOCK_SIZES
        1
        2
        4
        8
        13
        16
        32
    )
endif()
jacobi_generated_files(GKO_CUDA_JACOBI_SOURCES "${GKO_CUDA_JACOBI_BLOCK_SIZES}")
# override the default language mapping for the common files, set them to CUDA
foreach(
    source_file
    IN
    LISTS
        GKO_UNIFIED_COMMON_SOURCES
        GKO_CUDA_HIP_COMMON_SOURCES
        GKO_CUDA_JACOBI_SOURCES
        CSR_INSTANTIATE
        FBCSR_INSTANTIATE
        BATCH_BICGSTAB_INSTANTIATE1
        BATCH_BICGSTAB_INSTANTIATE2
        BATCH_CG_INSTANTIATE1
        BATCH_CG_INSTANTIATE2
)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
endforeach(source_file)
target_sources(ginkgo_cuda PRIVATE ${GKO_CUDA_JACOBI_SOURCES})
string(
    REPLACE
    ";"
    ","
    GKO_JACOBI_BLOCK_SIZES_CODE
    "${GKO_CUDA_JACOBI_BLOCK_SIZES}"
)
configure_file(
    ${Ginkgo_SOURCE_DIR}/common/cuda_hip/preconditioner/jacobi_common.hpp.in
    common/cuda_hip/preconditioner/jacobi_common.hpp
)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    # and allows the usage of std::array for nvidia GPUs
    target_compile_options(
        ginkgo_cuda
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    )
    if(MSVC)
        target_compile_options(
            ginkgo_cuda
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
        )
    else()
        target_compile_options(
            ginkgo_cuda
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        )
    endif()
endif()

ginkgo_compile_features(ginkgo_cuda)
target_compile_definitions(
    ginkgo_cuda
    PRIVATE GKO_COMPILING_CUDA GKO_DEVICE_NAMESPACE=cuda
)
if(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE)
    target_compile_definitions(
        ginkgo_cuda
        PRIVATE THRUST_CUB_WRAPPED_NAMESPACE=gko
    )
endif()

# include path for generated headers like jacobi_common.hpp
target_include_directories(ginkgo_cuda PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(
    ginkgo_cuda
    PRIVATE
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        CUDA::curand
        CUDA::cufft
        nvtx::nvtx
)
# NVTX3 is header-only and requires dlopen/dlclose in static builds
target_link_libraries(ginkgo_cuda PUBLIC ginkgo_device ${CMAKE_DL_LIBS})

ginkgo_default_includes(ginkgo_cuda)
ginkgo_install_library(ginkgo_cuda)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    set(check_header_def "GKO_COMPILING_CUDA;GKO_DEVICE_NAMESPACE=cuda")
    if(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE)
        set(check_header_def
            "${check_header_def};THRUST_CUB_WRAPPED_NAMESPACE=gko"
        )
    endif()
    ginkgo_check_headers(ginkgo_cuda "${check_header_def}")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
