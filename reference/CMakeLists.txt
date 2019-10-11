add_library(ginkgo_reference $<TARGET_OBJECTS:ginkgo_reference_device> "")
target_sources(ginkgo_reference
    PRIVATE
        base/version.cpp
        factorization/par_ilu_kernels.cpp
        matrix/coo_kernels.cpp
        matrix/csr_kernels.cpp
        matrix/dense_kernels.cpp
        matrix/ell_kernels.cpp
        matrix/hybrid_kernels.cpp
        matrix/sellp_kernels.cpp
        matrix/sparsity_csr_kernels.cpp
        preconditioner/jacobi_kernels.cpp
        solver/bicgstab_kernels.cpp
        solver/cg_kernels.cpp
        solver/cgs_kernels.cpp
        solver/fcg_kernels.cpp
        solver/gmres_kernels.cpp
        solver/ir_kernels.cpp
        solver/lower_trs_kernels.cpp
        solver/upper_trs_kernels.cpp
        stop/criterion_kernels.cpp
        stop/residual_norm_reduction_kernels.cpp)


ginkgo_compile_features(ginkgo_reference)
ginkgo_default_includes(ginkgo_reference)
ginkgo_install_library(ginkgo_reference reference)
target_compile_options(ginkgo_reference PRIVATE "${GINKGO_COMPILER_FLAGS}")

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
