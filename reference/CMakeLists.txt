set(SOURCES
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cpp
    matrix/dense_kernels.cpp
    matrix/ell_kernels.cpp
    preconditioner/block_jacobi_kernels.cpp
    solver/bicgstab_kernels.cpp
    solver/fcg_kernels.cpp
    solver/cg_kernels.cpp
    solver/tfqmr_kernels.cpp)

add_library(ginkgo_reference SHARED
    $<TARGET_OBJECTS:ginkgo_reference_device>
    ${SOURCES})
target_include_directories(ginkgo_reference PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
