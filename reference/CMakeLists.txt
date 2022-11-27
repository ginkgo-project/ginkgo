add_library(ginkgo_reference $<TARGET_OBJECTS:ginkgo_reference_device> "")
target_sources(ginkgo_reference
    PRIVATE
    base/device_matrix_data_kernels.cpp
    base/index_set_kernels.cpp
    base/scoped_device_id.cpp
    base/version.cpp
    components/absolute_array_kernels.cpp
    components/fill_array_kernels.cpp
    components/format_conversion_kernels.cpp
    components/precision_conversion_kernels.cpp
    components/prefix_sum_kernels.cpp
    components/reduce_array_kernels.cpp
    distributed/matrix_kernels.cpp
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
    factorization/par_ilut_kernels.cpp
    matrix/batch_csr_kernels.cpp
    matrix/batch_dense_kernels.cpp
    matrix/batch_diagonal_kernels.cpp
    matrix/batch_ell_kernels.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/diagonal_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/fbcsr_kernels.cpp
    matrix/fft_kernels.cpp
    matrix/hybrid_kernels.cpp
    matrix/sellp_kernels.cpp
    matrix/sparsity_csr_kernels.cpp
    multigrid/pgm_kernels.cpp
    preconditioner/batch_identity_kernels.cpp
    preconditioner/batch_ilu_isai_kernels.cpp
    preconditioner/batch_ilu_kernels.cpp
    preconditioner/batch_isai_kernels.cpp
    preconditioner/batch_jacobi_kernels.cpp
    preconditioner/isai_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    reorder/rcm_kernels.cpp
    solver/batch_bicgstab_kernels.cpp
    solver/batch_cg_kernels.cpp
    solver/batch_direct_kernels.cpp
    solver/batch_gmres_kernels.cpp
    solver/batch_idr_kernels.cpp
    solver/batch_lower_trs_kernels.cpp
    solver/batch_richardson_kernels.cpp
    solver/batch_upper_trs_kernels.cpp
    solver/bicg_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cb_gmres_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    solver/gmres_kernels.cpp
    solver/cb_gmres_kernels.cpp
    solver/common_gmres_kernels.cpp
    solver/idr_kernels.cpp
    solver/ir_kernels.cpp
    solver/lower_trs_kernels.cpp
    solver/multigrid_kernels.cpp
    solver/upper_trs_kernels.cpp
    stop/criterion_kernels.cpp
    stop/residual_norm_kernels.cpp)

target_link_libraries(ginkgo_reference PUBLIC ginkgo_device)
ginkgo_compile_features(ginkgo_reference)
ginkgo_default_includes(ginkgo_reference)
ginkgo_install_library(ginkgo_reference)
target_compile_options(ginkgo_reference PRIVATE "${GINKGO_COMPILER_FLAGS}")

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_reference "")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
