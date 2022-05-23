enable_language(CUDA)

# Needed because of a known issue with CUDA while linking statically.
# For details, see https://gitlab.kitware.com/cmake/cmake/issues/18614
if (NOT BUILD_SHARED_LIBS)
    set(CMAKE_CUDA_DEVICE_LINK_EXECUTABLE ${CMAKE_CUDA_DEVICE_LINK_EXECUTABLE} PARENT_SCOPE)
endif()

if(MSVC)
    # MSVC can not find CUDA automatically
    # Use CUDA_COMPILER PATH to define the CUDA TOOLKIT ROOT DIR
    string(REPLACE "/bin/nvcc.exe" "" CMAKE_CUDA_ROOT_DIR ${CMAKE_CUDA_COMPILER})
    if("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}" STREQUAL "")
        set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/include")
    endif()
    if("${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}" STREQUAL "")
        set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "${CMAKE_CUDA_ROOT_DIR}/lib/x64")
    endif()
endif()

include(CudaArchitectureSelector)

set(CUDA_INCLUDE_DIRS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# This is required by some examples such as custom_matrix_format
# which need the correct CMAKE_CUDA_FLAGS to be set
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" PARENT_SCOPE)
set(CMAKE_CUDA_COMPILER_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
set(CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS} PARENT_SCOPE)

# Detect the CUDA architecture flags and propagate to all the project
cas_variable_cuda_architectures(GINKGO_CUDA_ARCH_FLAGS
    ARCHITECTURES ${GINKGO_CUDA_ARCHITECTURES}
    UNSUPPORTED "20" "21")
set(GINKGO_CUDA_ARCH_FLAGS "${GINKGO_CUDA_ARCH_FLAGS}" PARENT_SCOPE)

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

# CUDA 10.1/10.2 put cublas, cublasLt, cudnn in /usr/lib/<arch>-linux-gnu/, but
# others (<= 10.0 or >= 11) put them in cuda own directory
# If the environment installs several cuda including 10.1/10.2, cmake will find
# the 10.1/10.2 .so files when searching others cuda in the default path.
# CMake already puts /usr/lib/<arch>-linux-gnu/ after cuda own directory in the
# `CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES`, so we always put NO_DEFAULT_PATH here.
find_library(CUBLAS cublas
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES} NO_DEFAULT_PATH)
find_library(CUSPARSE cusparse
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CURAND curand
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUFFT cufft
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_library(ginkgo_cuda $<TARGET_OBJECTS:ginkgo_cuda_device> "")
target_sources(ginkgo_cuda
    PRIVATE
    base/device_matrix_data_kernels.cu
    base/exception.cpp
    base/executor.cpp
    base/index_set_kernels.cpp
    base/version.cpp
    components/prefix_sum_kernels.cu
    distributed/matrix_kernels.cu
    distributed/partition_kernels.cu
    distributed/vector_kernels.cu
    factorization/cholesky_kernels.cu
    factorization/factorization_kernels.cu
    factorization/ic_kernels.cu
    factorization/ilu_kernels.cu
    factorization/par_ic_kernels.cu
    factorization/par_ict_kernels.cu
    factorization/par_ilu_kernels.cu
    factorization/par_ilut_approx_filter_kernel.cu
    factorization/par_ilut_filter_kernel.cu
    factorization/par_ilut_select_common.cu
    factorization/par_ilut_select_kernel.cu
    factorization/par_ilut_spgeam_kernel.cu
    factorization/par_ilut_sweep_kernel.cu
    matrix/coo_kernels.cu
    matrix/csr_kernels.cu
    matrix/dense_kernels.cu
    matrix/diagonal_kernels.cu
    matrix/ell_kernels.cu
    matrix/fbcsr_kernels.cu
    matrix/fft_kernels.cu
    matrix/sellp_kernels.cu
    matrix/sparsity_csr_kernels.cu
    multigrid/amgx_pgm_kernels.cu
    preconditioner/isai_kernels.cu
    preconditioner/jacobi_advanced_apply_kernel.cu
    preconditioner/jacobi_generate_kernel.cu
    preconditioner/jacobi_kernels.cu
    preconditioner/jacobi_simple_apply_kernel.cu
    reorder/rcm_kernels.cu
    solver/gmres_kernels.cu
    solver/cb_gmres_kernels.cu
    solver/idr_kernels.cu
    solver/lower_trs_kernels.cu
    solver/multigrid_kernels.cu
    solver/upper_trs_kernels.cu
    stop/criterion_kernels.cu
    stop/residual_norm_kernels.cu
    ${GKO_UNIFIED_COMMON_SOURCES}
    )
# override the default language mapping for the common files, set them to CUDA
foreach(source_file IN LISTS GKO_UNIFIED_COMMON_SOURCES)
    set_source_files_properties(${source_file} PROPERTIES LANGUAGE CUDA)
endforeach(source_file)
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_CUDA_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 32)
        list(APPEND GKO_CUDA_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_CUDA_JACOBI_BLOCK_SIZES 1 2 4 8 13 16 32)
endif()
set(GKO_CUDA_JACOBI_SOURCES)
foreach(GKO_JACOBI_BLOCK_SIZE IN LISTS GKO_CUDA_JACOBI_BLOCK_SIZES)
    configure_file(
        preconditioner/jacobi_generate_instantiate.inc.cu
        preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    configure_file(
        preconditioner/jacobi_simple_apply_instantiate.inc.cu
        preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    configure_file(
        preconditioner/jacobi_advanced_apply_instantiate.inc.cu
        preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
    list(APPEND GKO_CUDA_JACOBI_SOURCES
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.cu)
endforeach()
target_sources(ginkgo_cuda PRIVATE ${GKO_CUDA_JACOBI_SOURCES})
string(REPLACE ";" "," GKO_CUDA_JACOBI_BLOCK_SIZES_CODE "${GKO_CUDA_JACOBI_BLOCK_SIZES}")
configure_file(preconditioner/jacobi_common.hpp.in preconditioner/jacobi_common.hpp)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    # remove false positive CUDA warnings when calling one<T>() and zero<T>()
    # and allows the usage of std::array for nvidia GPUs
    target_compile_options(ginkgo_cuda
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    if(MSVC)
        target_compile_options(ginkgo_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    else()
        target_compile_options(ginkgo_cuda
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)
    endif()
endif()

if (NOT CMAKE_CUDA_HOST_COMPILER AND NOT GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE STRING "" FORCE)
elseif(GINKGO_CUDA_DEFAULT_HOST_COMPILER)
    unset(CMAKE_CUDA_HOST_COMPILER CACHE)
endif()

if(CMAKE_CUDA_HOST_COMPILER AND NOT CMAKE_CXX_COMPILER STREQUAL CMAKE_CUDA_HOST_COMPILER)
    message(WARNING "The CMake CXX compiler and CUDA host compiler do not match. "
        "If you encounter any build error, especially while linking, try to use "
        "the same compiler for both.\n"
        "The CXX compiler is ${CMAKE_CXX_COMPILER} with version ${CMAKE_CXX_COMPILER_VERSION}.\n"
        "The CUDA host compiler is ${CMAKE_CUDA_HOST_COMPILER}.")
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND CMAKE_CUDA_COMPILER_VERSION
    MATCHES "9.2" AND CMAKE_CUDA_HOST_COMPILER MATCHES ".*clang.*" )
    ginkgo_extract_clang_version(${CMAKE_CUDA_HOST_COMPILER} GINKGO_CUDA_HOST_CLANG_VERSION)

    if (GINKGO_CUDA_HOST_CLANG_VERSION MATCHES "5\.0.*")
        message(FATAL_ERROR "There is a bug between nvcc 9.2 and clang 5.0 which create a compiling issue."
            "Consider using a different CUDA host compiler or CUDA version.")
    endif()
endif()

target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>)
target_compile_options(ginkgo_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
ginkgo_compile_features(ginkgo_cuda)
target_compile_definitions(ginkgo_cuda PRIVATE GKO_COMPILING_CUDA)
target_include_directories(ginkgo_cuda
    SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# include path for generated headers like jacobi_common.hpp
target_include_directories(ginkgo_cuda
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)
target_link_libraries(ginkgo_cuda PRIVATE ${CUDA_RUNTIME_LIBS} ${CUBLAS} ${CUSPARSE} ${CURAND} ${CUFFT})
target_link_libraries(ginkgo_cuda PUBLIC ginkgo_device)
target_compile_options(ginkgo_cuda
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
# we handle CUDA architecture flags for now, disable CMake handling
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    set_target_properties(ginkgo_cuda PROPERTIES CUDA_ARCHITECTURES OFF)
endif()
list(GET CUDA_RUNTIME_LIBS 0 CUDA_FIRST_LIB)
get_filename_component(GKO_CUDA_LIBDIR "${CUDA_FIRST_LIB}" DIRECTORY)

ginkgo_default_includes(ginkgo_cuda)
ginkgo_install_library(ginkgo_cuda "${GKO_CUDA_LIBDIR}")

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_cuda GKO_COMPILING_CUDA)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Propagate some useful CUDA informations not propagated by default
set(CMAKE_CUDA_COMPILER_VERSION "${CMAKE_CUDA_COMPILER_VERSION}" PARENT_SCOPE)
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "${CMAKE_CUDA_HOST_LINK_LAUNCHER}" PARENT_SCOPE)
