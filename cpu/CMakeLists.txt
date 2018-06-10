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
    solver/fcg_kernels.cpp)

add_library(ginkgo_cpu
    $<TARGET_OBJECTS:ginkgo_cpu_device>
    ${SOURCES})
if(BUILD_GPU)
    target_link_libraries(ginkgo_cpu PUBLIC ginkgo_gpu)
endif()
ginkgo_default_includes(ginkgo_cpu)
ginkgo_install_library(ginkgo_cpu cpu)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
