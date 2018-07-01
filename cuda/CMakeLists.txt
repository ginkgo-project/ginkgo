set(SOURCES
    base/exception.cpp
    base/executor.cpp
    base/version.cpp
    matrix/coo_kernels.cu
    matrix/csr_kernels.cu
    matrix/dense_kernels.cu
    matrix/ell_kernels.cu
    matrix/hybrid_kernels.cpp
    preconditioner/block_jacobi_kernels.cu
    solver/bicgstab_kernels.cu
    solver/cg_kernels.cu
    solver/cgs_kernels.cu
    solver/fcg_kernels.cu
    stop/criterion_kernels.cu
    stop/residual_norm_reduction_kernels.cu)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUSPARSE cusparse
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(ginkgo_cuda
    $<TARGET_OBJECTS:ginkgo_cuda_device>
    ${SOURCES})
target_include_directories(ginkgo_cuda
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ginkgo_cuda PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS}
    ${CUSPARSE})
ginkgo_default_includes(ginkgo_cuda)
ginkgo_install_library(ginkgo_cuda cuda)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
