set(SOURCES
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/hybrid_kernels.cpp
    preconditioner/block_jacobi_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    stop/residual_norm_reduction_kernels.cpp)

add_library(ginkgo_omp
    $<TARGET_OBJECTS:ginkgo_omp_device>
    ${SOURCES})
target_link_libraries(ginkgo_omp PUBLIC ginkgo_gpu)
ginkgo_default_includes(ginkgo_omp)
ginkgo_install_library(ginkgo_omp omp)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
