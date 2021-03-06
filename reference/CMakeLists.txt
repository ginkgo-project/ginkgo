add_library(ginkgo_reference $<TARGET_OBJECTS:ginkgo_reference_device> "")
target_sources(ginkgo_reference
    PRIVATE
    base/version.cpp
    components/absolute_array.cpp
    components/fill_array.cpp
    components/precision_conversion.cpp
    components/prefix_sum.cpp
    factorization/factorization_kernels.cpp
    factorization/ic_kernels.cpp
    factorization/ilu_kernels.cpp
    factorization/par_ic_kernels.cpp
    factorization/par_ict_kernels.cpp
    factorization/par_ilu_kernels.cpp
    factorization/par_ilut_kernels.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/diagonal_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/fbcsr_kernels.cpp
    matrix/hybrid_kernels.cpp
    matrix/sellp_kernels.cpp
    matrix/sparsity_csr_kernels.cpp
    multigrid/amgx_pgm_kernels.cpp
    preconditioner/isai_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    reorder/rcm_kernels.cpp
    solver/bicg_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    solver/gmres_kernels.cpp
    solver/cb_gmres_kernels.cpp
    solver/idr_kernels.cpp
    solver/ir_kernels.cpp
    solver/lower_trs_kernels.cpp
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
