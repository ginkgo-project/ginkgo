set(SOURCES
    base/exception.cpp
    base/executor.cpp
    base/version.cpp
    matrix/coo_kernels.cu
    matrix/csr_kernels.cu
    matrix/dense_kernels.cu
    matrix/ell_kernels.cu
    matrix/hybrid_kernels.cpp
    matrix/sellp_kernels.cu
    preconditioner/block_jacobi_kernels.cu
    solver/bicgstab_kernels.cu
    solver/cg_kernels.cu
    solver/cgs_kernels.cu
    solver/fcg_kernels.cu
    stop/criterion_kernels.cu
    stop/residual_norm_reduction_kernels.cu)

enable_language(CUDA)
include(../cmake/detect_gpu.cmake)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(SET_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_FLAGS
        "${CMAKE_CUDA_FLAGS} --compiler-bindir ${CMAKE_CXX_COMPILER}")
endif()
# setting nvcc arch flags
ginkgo_select_nvcc_arch_flags("${CUDA_ARCHITECTURES}" NVCC_FLAGS_EXTRA)
message(STATUS "Added CUDA NVCC flags for: ${NVCC_FLAGS_EXTRA_readable}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA}")
# This creates a compilation bug on nvcc 9.0.102 *with* the new array_deleter
# merged at commit ed12b3df5d26
if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()

# make the CMAKE_CUDA_FLAGS in parent scope
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)

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
