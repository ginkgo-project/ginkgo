include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
set(CUDA_HIP_SOURCES
    base/batch_multi_vector_kernels.cpp
    base/device_matrix_data_kernels.cpp
    base/index_set_kernels.cpp
    components/prefix_sum_kernels.cpp
    distributed/index_map_kernels.cpp
    distributed/matrix_kernels.cpp
    distributed/partition_helpers_kernels.cpp
    distributed/partition_kernels.cpp
    distributed/vector_kernels.cpp
    factorization/cholesky_kernels.cpp
    factorization/factorization_kernels.cpp
    factorization/ic_kernels.cpp
    factorization/ilu_kernels.cpp
    factorization/lu_kernels.cpp
    factorization/par_ic_kernels.cpp
    factorization/par_ict_kernels.cpp
    factorization/par_ilu_kernels.cpp
    factorization/par_ilut_approx_filter_kernels.cpp
    factorization/par_ilut_filter_kernels.cpp
    factorization/par_ilut_select_common.cpp
    factorization/par_ilut_select_kernels.cpp
    factorization/par_ilut_spgeam_kernels.cpp
    factorization/par_ilut_sweep_kernels.cpp
    matrix/batch_csr_kernels.cpp
    matrix/batch_dense_kernels.cpp
    matrix/batch_ell_kernels.cpp
    matrix/coo_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/diagonal_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/sellp_kernels.cpp
    matrix/sparsity_csr_kernels.cpp
    multigrid/pgm_kernels.cpp
    preconditioner/isai_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    preconditioner/jacobi_advanced_apply_kernels.cpp
    preconditioner/jacobi_generate_kernels.cpp
    preconditioner/jacobi_simple_apply_kernels.cpp
    preconditioner/sor_kernels.cpp
    reorder/rcm_kernels.cpp
    solver/cb_gmres_kernels.cpp
    solver/idr_kernels.cpp
    solver/multigrid_kernels.cpp
    stop/criterion_kernels.cpp
    stop/residual_norm_kernels.cpp
    )
# create files for all potentially used block sizes
foreach(GKO_JACOBI_BLOCK_SIZE RANGE 1 64)
    configure_file(
        preconditioner/jacobi_generate_kernels.instantiate.cpp
        preconditioner/jacobi_generate_kernels.instantiate.${GKO_JACOBI_BLOCK_SIZE}.cpp)
    configure_file(
        preconditioner/jacobi_simple_apply_kernels.instantiate.cpp
        preconditioner/jacobi_simple_apply_kernels.instantiate.${GKO_JACOBI_BLOCK_SIZE}.cpp)
    configure_file(
        preconditioner/jacobi_advanced_apply_kernels.instantiate.cpp
        preconditioner/jacobi_advanced_apply_kernels.instantiate.${GKO_JACOBI_BLOCK_SIZE}.cpp)
endforeach()
function(jacobi_generated_files variable_name block_sizes)
    set(${variable_name})
    foreach(block_size IN LISTS block_sizes)
        list(APPEND variable_name
            ${Ginkgo_BINARY_DIR}/common/cuda_hip/preconditioner/jacobi_generate_kernels.instantiate.${block_size}.cpp
            ${Ginkgo_BINARY_DIR}/common/cuda_hip/preconditioner/jacobi_simple_apply_kernels.instantiate.${block_size}.cpp
            ${Ginkgo_BINARY_DIR}/common/cuda_hip/preconditioner/jacobi_advanced_apply_kernels.instantiate.${block_size}.cpp)
    endforeach()
    set(${variable_name} ${${variable_name}} PARENT_SCOPE)
endfunction()
list(TRANSFORM CUDA_HIP_SOURCES PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/)
set(GKO_CUDA_HIP_COMMON_SOURCES ${CUDA_HIP_SOURCES} PARENT_SCOPE)
