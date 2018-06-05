set(SOURCES
    base/exception.cpp
    base/executor.cpp
    matrix/coo_kernels.cpp
    matrix/csr_kernels.cu
    matrix/dense_kernels.cu
    matrix/ell_kernels.cu
    preconditioner/block_jacobi_kernels.cu
    solver/bicgstab_kernels.cu
    solver/cg_kernels.cu
    solver/cgs_kernels.cu
    solver/fcg_kernels.cu)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUSPARSE cusparse
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(ginkgo_gpu
    $<TARGET_OBJECTS:ginkgo_gpu_device>
    ${SOURCES})
target_include_directories(ginkgo_gpu PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(ginkgo_gpu
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ginkgo_gpu PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS}
  ${CUSPARSE})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
