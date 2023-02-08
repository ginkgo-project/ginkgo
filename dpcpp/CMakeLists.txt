find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}" "$ENV{MKL_ROOT}")
find_package(oneDPL REQUIRED HINTS "$ENV{DPL_ROOT}" "$ENV{DPLROOT}")
# use the parameter from cmake 
set(GINKGO_MKL_ROOT "${MKL_DIR}" PARENT_SCOPE)
set(GINKGO_DPL_ROOT "${oneDPL_DIR}" PARENT_SCOPE)

include(${PROJECT_SOURCE_DIR}/cmake/template_instantiation.cmake)
add_instantiation_files(${PROJECT_SOURCE_DIR}/common/unified matrix/dense_kernels.instantiate.cpp DENSE_INSTANTIATE)
add_library(ginkgo_dpcpp $<TARGET_OBJECTS:ginkgo_dpcpp_device> "")
target_sources(ginkgo_dpcpp
    PRIVATE
    base/batch_multi_vector_kernels.dp.cpp
    base/device_matrix_data_kernels.dp.cpp
    base/executor.dp.cpp
    base/helper.dp.cpp
    base/index_set_kernels.dp.cpp
    base/scoped_device_id.dp.cpp
    base/timer.dp.cpp
    base/version.dp.cpp
    components/prefix_sum_kernels.dp.cpp
    distributed/matrix_kernels.dp.cpp
    distributed/partition_helpers_kernels.dp.cpp
    distributed/partition_kernels.dp.cpp
    distributed/vector_kernels.dp.cpp
    factorization/cholesky_kernels.dp.cpp
    factorization/ic_kernels.dp.cpp
    factorization/ilu_kernels.dp.cpp
    factorization/factorization_kernels.dp.cpp
    factorization/lu_kernels.dp.cpp
    factorization/par_ic_kernels.dp.cpp
    factorization/par_ict_kernels.dp.cpp
    factorization/par_ilu_kernels.dp.cpp
    factorization/par_ilut_approx_filter_kernel.dp.cpp
    factorization/par_ilut_filter_kernel.dp.cpp
    factorization/par_ilut_select_common.dp.cpp
    factorization/par_ilut_select_kernel.dp.cpp
    factorization/par_ilut_spgeam_kernel.dp.cpp
    factorization/par_ilut_sweep_kernel.dp.cpp
    matrix/batch_csr_kernels.dp.cpp
    matrix/batch_dense_kernels.dp.cpp
    matrix/batch_ell_kernels.dp.cpp
    matrix/coo_kernels.dp.cpp
    matrix/csr_kernels.dp.cpp
    matrix/fbcsr_kernels.dp.cpp
    matrix/dense_kernels.dp.cpp
    matrix/diagonal_kernels.dp.cpp
    matrix/ell_kernels.dp.cpp
    matrix/fft_kernels.dp.cpp
    matrix/sellp_kernels.dp.cpp
    matrix/sparsity_csr_kernels.dp.cpp
    multigrid/pgm_kernels.dp.cpp
    preconditioner/isai_kernels.dp.cpp
    preconditioner/jacobi_advanced_apply_kernel.dp.cpp
    preconditioner/jacobi_generate_kernel.dp.cpp
    preconditioner/jacobi_kernels.dp.cpp
    preconditioner/jacobi_simple_apply_kernel.dp.cpp
    reorder/rcm_kernels.dp.cpp
    solver/batch_bicgstab_kernels.dp.cpp
    solver/cb_gmres_kernels.dp.cpp
    solver/idr_kernels.dp.cpp
    solver/lower_trs_kernels.dp.cpp
    solver/multigrid_kernels.dp.cpp
    solver/upper_trs_kernels.dp.cpp
    stop/criterion_kernels.dp.cpp
    stop/residual_norm_kernels.dp.cpp
    ${GKO_UNIFIED_COMMON_SOURCES}
    ${DENSE_INSTANTIATE}
    )

# TODO: adjust it when dpcpp jacobi supports more block size
set(GKO_DPCPP_JACOBI_BLOCK_SIZES 32)
set(GKO_DPCPP_JACOBI_SOURCES)
foreach(GKO_JACOBI_BLOCK_SIZE IN LISTS GKO_DPCPP_JACOBI_BLOCK_SIZES)
    configure_file(
        preconditioner/jacobi_generate_instantiate.inc.dp.cpp
        preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp)
    configure_file(
        preconditioner/jacobi_simple_apply_instantiate.inc.dp.cpp
        preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp)
    configure_file(
        preconditioner/jacobi_advanced_apply_instantiate.inc.dp.cpp
        preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp)
    list(APPEND GKO_DPCPP_JACOBI_SOURCES
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_generate_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_simple_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/preconditioner/jacobi_advanced_apply_instantiate.${GKO_JACOBI_BLOCK_SIZE}.dp.cpp)
endforeach()
target_sources(ginkgo_dpcpp PRIVATE ${GKO_DPCPP_JACOBI_SOURCES})
string(REPLACE ";" "," GKO_DPCPP_JACOBI_BLOCK_SIZES_CODE "${GKO_DPCPP_JACOBI_BLOCK_SIZES}")
configure_file(preconditioner/jacobi_common.hpp.in preconditioner/jacobi_common.hpp)

ginkgo_compile_features(ginkgo_dpcpp)
target_compile_definitions(ginkgo_dpcpp PRIVATE GKO_COMPILING_DPCPP _ONEDPL_COMPILE_KERNEL=0)

set(GINKGO_DPCPP_FLAGS ${GINKGO_DPCPP_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_DPCPP_FLAGS}")
# all file in target ginkgo_dpcpp are necessarily compiled with sycl, so we can ignore the warning.
# If we would like to use SOURCES, please use the new files copied from GKO_UNIFIED_COMMON_SOURCES.
# Otherwise, the source's properties will be changed by add_sycl_to_target
gko_add_sycl_to_target(TARGET ginkgo_dpcpp)
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_COMPILER_FLAGS}")
# Note: add MKL as PRIVATE not PUBLIC (MKL example shows) to avoid propagating
# find_package(MKL) everywhere when linking ginkgo (see the MKL example
# https://software.intel.com/content/www/us/en/develop/documentation/onemkl-windows-developer-guide/top/getting-started/cmake-config-for-onemkl.html)
target_compile_features(ginkgo_dpcpp PUBLIC cxx_std_17)
target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)
# When building ginkgo as a static library, we need to use dpcpp and per_kernel
# link option when the program uses a dpcpp related function.
if (BUILD_SHARED_LIBS)
    target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-code-split=per_kernel)
else ()
    target_link_options(ginkgo_dpcpp PUBLIC -fsycl-device-code-split=per_kernel)
endif()
# include path for generated headers like jacobi_common.hpp
target_include_directories(ginkgo_dpcpp
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/..)
target_link_libraries(ginkgo_dpcpp PUBLIC ginkgo_device)
target_link_libraries(ginkgo_dpcpp PRIVATE MKL::MKL_DPCPP oneDPL)
if (GINKGO_DPCPP_SINGLE_MODE)
    target_compile_definitions(ginkgo_dpcpp PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
endif()

ginkgo_default_includes(ginkgo_dpcpp)
ginkgo_install_library(ginkgo_dpcpp)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_dpcpp GKO_COMPILING_DPCPP)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
