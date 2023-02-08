include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(. matrix/csr_kernels.instantiate.hip.cpp CSR_INSTANTIATE)
add_instantiation_files(. matrix/fbcsr_kernels.instantiate.hip.cpp FBCSR_INSTANTIATE)
# we don't split up the dense kernels into distinct compilations
list(APPEND GKO_UNIFIED_COMMON_SOURCES ${PROJECT_SOURCE_DIR}/common/unified/matrix/dense_kernels.instantiate.cpp)
set(GINKGO_HIP_SOURCES
    base/batch_multi_vector_kernels.hip.cpp
    base/device.hip.cpp
    base/device_matrix_data_kernels.hip.cpp
    base/exception.hip.cpp
    base/executor.hip.cpp
    base/index_set_kernels.hip.cpp
    base/memory.hip.cpp
    base/roctx.hip.cpp
    base/scoped_device_id.hip.cpp
    base/stream.hip.cpp
    base/timer.hip.cpp
    base/version.hip.cpp
    components/prefix_sum_kernels.hip.cpp
    distributed/matrix_kernels.hip.cpp
    distributed/partition_helpers_kernels.hip.cpp
    distributed/partition_kernels.hip.cpp
    distributed/vector_kernels.hip.cpp
    factorization/cholesky_kernels.hip.cpp
    factorization/factorization_kernels.hip.cpp
    factorization/ic_kernels.hip.cpp
    factorization/ilu_kernels.hip.cpp
    factorization/lu_kernels.hip.cpp
    factorization/par_ic_kernels.hip.cpp
    factorization/par_ict_kernels.hip.cpp
    factorization/par_ilu_kernels.hip.cpp
    factorization/par_ilut_approx_filter_kernel.hip.cpp
    factorization/par_ilut_filter_kernel.hip.cpp
    factorization/par_ilut_select_common.hip.cpp
    factorization/par_ilut_select_kernel.hip.cpp
    factorization/par_ilut_spgeam_kernel.hip.cpp
    factorization/par_ilut_sweep_kernel.hip.cpp
    matrix/batch_csr_kernels.hip.cpp
    matrix/batch_dense_kernels.hip.cpp
    matrix/batch_ell_kernels.hip.cpp
    matrix/coo_kernels.hip.cpp
    ${CSR_INSTANTIATE}
    matrix/dense_kernels.hip.cpp
    matrix/diagonal_kernels.hip.cpp
    matrix/ell_kernels.hip.cpp
    ${FBCSR_INSTANTIATE}
    matrix/sellp_kernels.hip.cpp
    matrix/sparsity_csr_kernels.hip.cpp
    multigrid/pgm_kernels.hip.cpp
    preconditioner/isai_kernels.hip.cpp
    preconditioner/jacobi_advanced_apply_kernel.hip.cpp
    preconditioner/jacobi_generate_kernel.hip.cpp
    preconditioner/jacobi_kernels.hip.cpp
    preconditioner/jacobi_simple_apply_kernel.hip.cpp
    reorder/rcm_kernels.hip.cpp
    solver/batch_bicgstab_kernels.hip.cpp
    solver/cb_gmres_kernels.hip.cpp
    solver/idr_kernels.hip.cpp
    solver/lower_trs_kernels.hip.cpp
    solver/multigrid_kernels.hip.cpp
    solver/upper_trs_kernels.hip.cpp
    stop/criterion_kernels.hip.cpp
    stop/residual_norm_kernels.hip.cpp
    ${GKO_UNIFIED_COMMON_SOURCES}
    )

if(hipfft_FOUND)
    list(APPEND GINKGO_HIP_SOURCES matrix/fft_kernels.hip.cpp)
else()
    list(APPEND GINKGO_HIP_SOURCES matrix/fft_kernels_stub.hip.cpp)
endif()

if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
    set(GKO_HIP_JACOBI_MAX_BLOCK_SIZE 32)
else()
    set(GKO_HIP_JACOBI_MAX_BLOCK_SIZE 64)
endif()
if(GINKGO_JACOBI_FULL_OPTIMIZATIONS)
    set(GKO_HIP_JACOBI_BLOCK_SIZES)
    foreach(blocksize RANGE 1 ${GKO_HIP_JACOBI_MAX_BLOCK_SIZE})
        list(APPEND GKO_HIP_JACOBI_BLOCK_SIZES ${blocksize})
    endforeach()
else()
    set(GKO_HIP_JACOBI_BLOCK_SIZES 1 2 4 8 13 16 32 ${GKO_HIP_JACOBI_MAX_BLOCK_SIZE})
    list(REMOVE_DUPLICATES GKO_HIP_JACOBI_BLOCK_SIZES)
endif()
foreach(GKO_JACOBI_BLOCK_SIZE IN LISTS GKO_HIP_JACOBI_BLOCK_SIZES)
    configure_file(
        preconditioner/jacobi_generate_instantiate.inc.hip.cpp
        preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp)
    configure_file(
        preconditioner/jacobi_simple_apply_instantiate.inc.hip.cpp
        preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp)
    configure_file(
        preconditioner/jacobi_advanced_apply_instantiate.inc.hip.cpp
        preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp)
    list(APPEND GINKGO_HIP_SOURCES
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.hip.cpp)
endforeach()
string(REPLACE ";" "," GKO_HIP_JACOBI_BLOCK_SIZES_CODE "${GKO_HIP_JACOBI_BLOCK_SIZES}")
configure_file(preconditioner/jacobi_common.hip.hpp.in preconditioner/jacobi_common.hip.hpp)

set_source_files_properties(${GINKGO_HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
hip_add_library(ginkgo_hip $<TARGET_OBJECTS:ginkgo_hip_device> ${GINKGO_HIP_SOURCES}
    HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
    CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
    NVCC_OPTIONS ${GINKGO_HIP_NVCC_OPTIONS}
    ${GINKGO_STATIC_OR_SHARED})

target_include_directories(ginkgo_hip
    PUBLIC
        ${HIP_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_BINARY_DIR}/.. # for generated headers like jacobi_common.hip.hpp
        ${GINKGO_HIP_THRUST_PATH}
        ${HIPBLAS_INCLUDE_DIRS}
        ${HIPFFT_INCLUDE_DIRS}
        ${hiprand_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        $<BUILD_INTERFACE:${ROCPRIM_INCLUDE_DIRS}>)

target_link_libraries(ginkgo_hip PUBLIC ginkgo_device)
target_link_libraries(ginkgo_hip PRIVATE roc::hipblas roc::hipsparse hip::hiprand roc::rocrand)
if (hipfft_FOUND)
    target_link_libraries(ginkgo_hip PRIVATE hip::hipfft)
endif()
if (GINKGO_HAVE_ROCTX)
    target_link_libraries(ginkgo_hip PRIVATE roc::roctx)
endif()

target_compile_options(ginkgo_hip PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)

if(GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    find_package(hip REQUIRED)
    # To save a bit of pain, we directly link against the `library` instead of
    # linking against the target.
    if (CMAKE_BUILD_TYPE)
        # Check if our configuration is available first
        string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
        get_target_property(HIP_LIBAMDHIP64_LIBRARIES hip::amdhip64 IMPORTED_LOCATION_${UPPER_BUILD_TYPE})
    endif()
    if (NOT HIP_LIBAMDHIP64_LIBRARIES)
        # Fall back to anything
        get_target_property(HIP_LIBAMDHIP64_LIBRARIES hip::amdhip64 LOCATION)
    endif()
    target_link_libraries(ginkgo_hip PUBLIC ${HIP_LIBAMDHIP64_LIBRARIES})
elseif(GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
    find_package(CUDA 10.1 REQUIRED)
    target_link_libraries(ginkgo_hip PUBLIC ${CUDA_LIBRARIES})
endif()

# Try to find everything in /opt/rocm/lib first.
set(GKO_HIP_RPATH "${ROCM_PATH}/lib" )
if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
    list(GET CUDA_LIBRARIES 0 CUDA_FIRST_LIB)
    get_filename_component(GKO_CUDA_LIBDIR "${CUDA_FIRST_LIB}" DIRECTORY)
    list(APPEND GKO_HIP_RPATH "${GKO_CUDA_LIBDIR}")
else()
    list(APPEND GKO_HIP_RPATH "${HIP_PATH}/lib")
endif()
list(APPEND GKO_HIP_RPATH "${HIPBLAS_PATH}/lib" "${HIPRAND_PATH}/lib"
    "${HIPSPARSE_PATH}/lib" "${ROCRAND_PATH}/lib")

ginkgo_compile_features(ginkgo_hip)
ginkgo_default_includes(ginkgo_hip)
ginkgo_install_library(ginkgo_hip "${GKO_HIP_RPATH}")

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_hip GKO_COMPILING_HIP)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
