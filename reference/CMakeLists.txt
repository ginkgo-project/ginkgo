set(SOURCES
    base/version.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/csri_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/hybrid_kernels.cpp
    matrix/sellp_kernels.cpp
    preconditioner/block_jacobi_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    solver/gmres_kernels.cpp
    stop/criterion_kernels.cpp
    stop/residual_norm_reduction_kernels.cpp)

add_library(ginkgo_reference
    $<TARGET_OBJECTS:ginkgo_reference_device>
    ${SOURCES})
ginkgo_default_includes(ginkgo_reference)
ginkgo_install_library(ginkgo_reference reference)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
