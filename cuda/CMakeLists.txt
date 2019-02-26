enable_language(CUDA)
include(CudaArchitectureSelector)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# This is required by some examples such as cmake_matrix_format
# which need the correct CMAKE_CUDA_FLAGS to be set
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
set(CUDA_VERSION ${CUDA_VERSION} PARENT_SCOPE)
set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)

find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUSPARSE cusparse
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(ginkgo_cuda $<TARGET_OBJECTS:ginkgo_cuda_device> "")
target_sources(ginkgo_cuda
    PRIVATE
        base/exception.cpp
        base/executor.cpp
        base/version.cpp
        components/zero_array.cu
        matrix/coo_kernels.cu
        matrix/csr_kernels.cu
        matrix/dense_kernels.cu
        matrix/ell_kernels.cu
        matrix/hybrid_kernels.cpp
        matrix/sellp_kernels.cu
        preconditioner/jacobi_kernels.cu
        solver/bicgstab_kernels.cu
        solver/cg_kernels.cu
        solver/cgs_kernels.cu
        solver/fcg_kernels.cu
        solver/gmres_kernels.cu
        solver/ir_kernels.cu
        stop/criterion_kernels.cu
        stop/residual_norm_reduction_kernels.cu)

# This creates a compilation bug on nvcc 9.0.102 *with* the new array_deleter
# merged at commit ed12b3df5d26
if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "9.0")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    target_compile_options(ginkgo_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

ginkgo_compile_features(ginkgo_cuda)
target_include_directories(ginkgo_cuda
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ginkgo_cuda PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS} ${CUSPARSE})

# Ensures that there are no circular dependency issues
if(NOT BUILD_SHARED_LIBS)
    target_link_libraries(ginkgo_cuda PUBLIC ginkgo)
endif()

cas_target_cuda_architectures(ginkgo_cuda
    ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
    UNSUPPORTED "20" "21")

ginkgo_default_includes(ginkgo_cuda)
ginkgo_install_library(ginkgo_cuda cuda)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Propagate some useful CUDA informations not propagated by default
set(CMAKE_CUDA_COMPILER_VERSION "${CMAKE_CUDA_COMPILER_VERSION}" PARENT_SCOPE)
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CUDA_HOST_LINK_LAUNCHER}" PARENT_SCOPE)
