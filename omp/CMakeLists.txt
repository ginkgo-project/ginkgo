set(SOURCES
    base/version.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/ell_kernels.cpp
    matrix/hybrid_kernels.cpp
    matrix/sellp_kernels.cpp
    preconditioner/jacobi_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/cg_kernels.cpp
    solver/cgs_kernels.cpp
    solver/fcg_kernels.cpp
    solver/gmres_kernels.cpp
    stop/criterion_kernels.cpp
    stop/residual_norm_reduction_kernels.cpp)

find_package(OpenMP REQUIRED)

add_library(ginkgo_omp
    $<TARGET_OBJECTS:ginkgo_omp_device>
    ${SOURCES})

target_link_libraries(ginkgo_omp PRIVATE "${OpenMP_CXX_FLAGS}")
target_compile_options(ginkgo_omp PRIVATE "${OpenMP_CXX_FLAGS}")

# Ensures that there are no circular dependency issues
if(NOT BUILD_SHARED_LIBS)
    target_link_libraries(ginkgo_omp PUBLIC ginkgo)
endif()

ginkgo_default_includes(ginkgo_omp)
ginkgo_install_library(ginkgo_omp omp)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
