find_package(OpenMP REQUIRED)

add_library(ginkgo_omp $<TARGET_OBJECTS:ginkgo_omp_device> "")
target_sources(ginkgo_omp
    PRIVATE
        base/version.cpp
        factorization/par_ilu_kernels.cpp
        matrix/coo_kernels.cpp
        matrix/csr_kernels.cpp
        matrix/dense_kernels.cpp
        matrix/ell_kernels.cpp
        matrix/hybrid_kernels.cpp
        matrix/sellp_kernels.cpp
        matrix/sparsity_kernels.cpp
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

ginkgo_compile_features(ginkgo_omp)
target_link_libraries(ginkgo_omp PRIVATE "${OpenMP_CXX_LIBRARIES}")
target_compile_options(ginkgo_omp PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(ginkgo_omp PRIVATE "${GINKGO_COMPILER_FLAGS}")

# Need to link against ginkgo_cuda for the `raw_copy_to(CudaExecutor ...)` method
target_link_libraries(ginkgo_omp PUBLIC ginkgo_cuda)

ginkgo_default_includes(ginkgo_omp)
ginkgo_install_library(ginkgo_omp omp)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Propagate some useful information
set(OpenMP_CXX_VERSION ${OpenMP_CXX_VERSION} PARENT_SCOPE)
set(OpenMP_CXX_LIBRARIES ${OpenMP_CXX_LIBRARIES} PARENT_SCOPE)
