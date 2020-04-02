enable_language(CUDA)

# Needed because of a known issue with CUDA while linking statically.
# For details, see https://gitlab.kitware.com/cmake/cmake/issues/18614
if (NOT BUILD_SHARED_LIBS)
    set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE ${CMAKE_CUDA_DEVICE_LINK_EXECUTABLE} PARENT_SCOPE)
endif()

if(MSVC)
    # MSVC can not find CUDA automatically
    # Use CUDA_COMPILER PATH to define the CUDA TOOLKIT ROOT DIR
    if("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" STREQUAL "")
        string(REPLACE "/bin/nvcc.exe" "" CMAKE_CUDA_ROOT_DIR ${CMAKE_CUDA_COMPILER})
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/include")
        set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/lib/x64")
    endif()

    # This is modified from https://gitlab.kitware.com/cmake/community/wikis/FAQ#dynamic-replace
    if(BUILD_SHARED_LIBS)
        ginkgo_switch_to_windows_dynamic("CUDA")
    else()
        ginkgo_switch_to_windows_static("CUDA")
    endif()
endif()

include(CudaArchitectureSelector)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# This is required by some examples such as cmake_matrix_format
# which need the correct CMAKE_CUDA_FLAGS to be set
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
set(CMAKE_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)


# MSVC nvcc uses static cudartlibrary by default, and other platforms use shared cudartlibrary.
# add `-cudart shared` or `-cudart=shared` according system into CMAKE_CUDA_FLAGS
# to force nvcc to use dynamic cudart library in MSVC.
find_library(CUDA_RUNTIME_LIBS_DYNAMIC cudart
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_RUNTIME_LIBS_STATIC cudart_static
        HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
if(MSVC)
    if("${CMAKE_CUDA_FLAGS}" MATCHES "-cudart(=| )shared")
        set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_DYNAMIC}" CACHE STRING "Path to a library" FORCE)
    else()
        set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_STATIC}" CACHE STRING "Path to a library" FORCE)
    endif()
else()
    set(CUDA_RUNTIME_LIBS "${CUDA_RUNTIME_LIBS_DYNAMIC}" CACHE STRING "Path to a library" FORCE)
endif()

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
        components/prefix_sum.cu
        components/zero_array_kernels.cu
        factorization/ilu_kernels.cu
        factorization/factorization_kernels.cu
        factorization/par_ilu_kernels.cu
        matrix/coo_kernels.cu
        matrix/csr_kernels.cu
        matrix/dense_kernels.cu
        matrix/ell_kernels.cu
        matrix/hybrid_kernels.cu
        matrix/sellp_kernels.cu
        matrix/sparsity_csr_kernels.cu
        preconditioner/jacobi_advanced_apply_kernel.cu
        preconditioner/jacobi_generate_kernel.cu
        preconditioner/jacobi_kernels.cu
        preconditioner/jacobi_simple_apply_kernel.cu
        solver/bicg_kernels.cu
        solver/bicgstab_kernels.cu
        solver/cg_kernels.cu
        solver/cgs_kernels.cu
        solver/fcg_kernels.cu
        solver/gmres_kernels.cu
        solver/ir_kernels.cu
        solver/lower_trs_kernels.cu
        solver/upper_trs_kernels.cu
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

if (NOT CMAKE_CUDA_HOST_COMPILER AND NOT GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
elseif(GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
endif()
target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>)
target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
ginkgo_compile_features(ginkgo_cuda)
target_include_directories(ginkgo_cuda
    SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(ginkgo_cuda PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS} ${CUSPARSE})

# Need to link against ginkgo_hip for the `raw_copy_to(HipExecutor ...)` method
target_link_libraries(ginkgo_cuda PUBLIC ginkgo_hip)

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
