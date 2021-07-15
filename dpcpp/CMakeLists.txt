if (NOT GKO_CAN_COMPILE_DPCPP)
    message(FATAL_ERROR "The CMAKE_CXX_COMPILER compiler, which is "
        "${CMAKE_CXX_COMPILER} cannot compile DPC++ code!")
endif()

ginkgo_extract_dpcpp_version(${CMAKE_CXX_COMPILER} GINKGO_DPCPP_VERSION)
set(GINKGO_DPCPP_VERSION ${GINKGO_DPCPP_VERSION} PARENT_SCOPE)

add_library(ginkgo_dpcpp $<TARGET_OBJECTS:ginkgo_dpcpp_device> "")
target_sources(ginkgo_dpcpp
    PRIVATE
    base/version.dp.cpp
    base/executor.dp.cpp
    components/absolute_array.dp.cpp
    components/fill_array.dp.cpp
    components/prefix_sum.dp.cpp
    factorization/ic_kernels.dp.cpp
    factorization/ilu_kernels.dp.cpp
    factorization/factorization_kernels.dp.cpp
    factorization/par_ic_kernels.dp.cpp
    factorization/par_ict_kernels.dp.cpp
    factorization/par_ilu_kernels.dp.cpp
    factorization/par_ilut_kernels.dp.cpp
    matrix/coo_kernels.dp.cpp
    matrix/csr_kernels.dp.cpp
    matrix/fbcsr_kernels.dp.cpp
    matrix/dense_kernels.dp.cpp
    matrix/diagonal_kernels.dp.cpp
    matrix/ell_kernels.dp.cpp
    matrix/hybrid_kernels.dp.cpp
    matrix/sellp_kernels.dp.cpp
    matrix/sparsity_csr_kernels.dp.cpp
    multigrid/amgx_pgm_kernels.dp.cpp
    preconditioner/isai_kernels.dp.cpp
    preconditioner/jacobi_kernels.dp.cpp
    reorder/rcm_kernels.dp.cpp
    solver/gmres_kernels.dp.cpp
    solver/cb_gmres_kernels.dp.cpp
    solver/idr_kernels.dp.cpp
    solver/lower_trs_kernels.dp.cpp
    solver/multigrid_kernels.dp.cpp
    solver/upper_trs_kernels.dp.cpp
    stop/criterion_kernels.dp.cpp
    stop/residual_norm_kernels.dp.cpp
    ../common/components/precision_conversion.cpp
    ../common/matrix/dense_kernels.cpp
    ../common/solver/bicg_kernels.cpp
    ../common/solver/bicgstab_kernels.cpp
    ../common/solver/cg_kernels.cpp
    ../common/solver/cgs_kernels.cpp
    ../common/solver/fcg_kernels.cpp
    ../common/solver/ir_kernels.cpp
    )

ginkgo_compile_features(ginkgo_dpcpp)
target_compile_definitions(ginkgo_dpcpp PRIVATE GKO_COMPILING_DPCPP)

set(GINKGO_DPCPP_FLAGS ${GINKGO_COMPILER_FLAGS} -fsycl)
set(GINKGO_DPCPP_FLAGS ${GINKGO_DPCPP_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_DPCPP_FLAGS}")
target_compile_features(ginkgo_dpcpp PRIVATE cxx_std_17)
target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)
target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-code-split=per_kernel)
target_link_libraries(ginkgo_dpcpp PUBLIC ginkgo_device)
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
