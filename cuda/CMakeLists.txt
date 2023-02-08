cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
add_library(ginkgo_cuda $<TARGET_OBJECTS:ginkgo_cuda_device> "")
include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(. matrix/csr_kernels.instantiate.cu CSR_INSTANTIATE)
add_instantiation_files(. matrix/fbcsr_kernels.instantiate.cu FBCSR_INSTANTIATE)
# we don't split up the dense kernels into distinct compilations
list(APPEND GKO_UNIFIED_COMMON_SOURCES ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp)
target_sources(ginkgo_cuda
    PRIVATE
    base/batch_multi_vector_kernels.cu
    base/device.cpp
    base/device_matrix_data_kernels.cu
    base/exception.cpp
    base/executor.cpp
    base/index_set_kernels.cpp
    base/memory.cpp
    base/nvtx.cpp
    base/scoped_device_id.cpp
    base/stream.cpp
    base/timer.cpp
    base/version.cpp
    components/prefix_sum_kernels.cu
    distributed/matrix_kernels.cu
    distributed/partition_helpers_kernels.cu
    distributed/partition_kernels.cu
    distributed/vector_kernels.cu
    factorization/cholesky_kernels.cu
    factorization/factorization_kernels.cu
    factorization/ic_kernels.cu
    factorization/ilu_kernels.cu
    factorization/lu_kernels.cu
    factorization/par_ic_kernels.cu
    factorization/par_ict_kernels.cu
    factorization/par_ilu_kernels.cu
    factorization/par_ilut_approx_filter_kernel.cu
    factorization/par_ilut_filter_kernel.cu
    factorization/par_ilut_select_common.cu
    factorization/par_ilut_select_kernel.cu
    factorization/par_ilut_spgeam_kernel.cu
    factorization/par_ilut_sweep_kernel.cu
    matrix/batch_csr_kernels.cu
    matrix/batch_dense_kernels.cu
    matrix/batch_ell_kernels.cu
    matrix/coo_kernels.cu
    ${CSR_INSTANTIATE}
    matrix/dense_kernels.cu
    matrix/diagonal_kernels.cu
    matrix/ell_kernels.cu
    ${FBCSR_INSTANTIATE}
    matrix/fft_kernels.cu
    matrix/sellp_kernels.cu
    matrix/sparsity_csr_kernels.cu
    multigrid/pgm_kernels.cu
    preconditioner/isai_kernels.cu
    preconditioner/jacobi_advanced_apply_kernel.cu
    preconditioner/jacobi_generate_kernel.cu
    preconditioner/jacobi_kernels.cu
    preconditioner/jacobi_simple_apply_kernel.cu
    reorder/rcm_kernels.cu
    solver/batch_bicgstab_kernels.cu
    solver/cb_gmres_kernels.cu
    solver/idr_kernels.cu
    solver/lower_trs_kernels.cu
    solver/multigrid_kernels.cu
    solver/upper_trs_kernels.cu
    stop/criterion_kernels.cu
    stop/residual_norm_kernels.cu
    ${GKO_UNIFIED_COMMON_SOURCES}
    )
# override the default language mapping for the common files, set them to CUDA
foreach(source_file IN LISTS GKO_UNIFIED_COMMON_SOURCES)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
endforeach(source_file)
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_CUDA_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 32)
        list(APPEND GKO_CUDA_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_CUDA_JACOBI_BLOCK_SIZES 1 2 4 8 13 16 32)
endif()
set(GKO_CUDA_JACOBI_SOURCES)
foreach(GKO_JACOBI_BLOCK_SIZE IN LISTS GKO_CUDA_JACOBI_BLOCK_SIZES)
    configure_file(
        preconditioner/jacobi_generate_instantiate.inc.cu
        preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    configure_file(
        preconditioner/jacobi_simple_apply_instantiate.inc.cu
        preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    configure_file(
        preconditioner/jacobi_advanced_apply_instantiate.inc.cu
        preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    list(APPEND GKO_CUDA_JACOBI_SOURCES
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
endforeach()
target_sources(ginkgo_cuda PRIVATE ${GKO_CUDA_JACOBI_SOURCES})
string(REPLACE ";" "," GKO_CUDA_JACOBI_BLOCK_SIZES_CODE "${GKO_CUDA_JACOBI_BLOCK_SIZES}")
configure_file(preconditioner/jacobi_common.hpp.in preconditioner/jacobi_common.hpp)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    # and allows the usage of std::array for nvidia GPUs
    target_compile_options(ginkgo_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(MSVC)
        target_compile_options(ginkgo_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    else()
        target_compile_options(ginkgo_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    endif()
endif()

target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>)
target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
ginkgo_compile_features(ginkgo_cuda)
target_compile_definitions(ginkgo_cuda PRIVATE GKO_COMPILING_CUDA)

# include path for generated headers like jacobi_common.hpp
target_include_directories(ginkgo_cuda
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)
target_link_libraries(ginkgo_cuda PRIVATE CUDA::cudart CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cufft nvtx::nvtx)
# NVTX3 is header-only and requires dlopen/dlclose in static builds
target_link_libraries(ginkgo_cuda PUBLIC ginkgo_device ${CMAKE_DL_LIBS})

ginkgo_default_includes(ginkgo_cuda)
ginkgo_install_library(ginkgo_cuda)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_cuda GKO_COMPILING_CUDA)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
