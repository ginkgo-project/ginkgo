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
    components/precision_conversion.dp.cpp
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
    matrix/dense_kernels.dp.cpp
    matrix/diagonal_kernels.dp.cpp
    matrix/ell_kernels.dp.cpp
    matrix/hybrid_kernels.dp.cpp
    matrix/sellp_kernels.dp.cpp
    matrix/sparsity_csr_kernels.dp.cpp
    preconditioner/isai_kernels.dp.cpp
    preconditioner/jacobi_kernels.dp.cpp
    reorder/rcm_kernels.dp.cpp
    solver/bicg_kernels.dp.cpp
    solver/bicgstab_kernels.dp.cpp
    solver/cg_kernels.dp.cpp
    solver/cgs_kernels.dp.cpp
    solver/fcg_kernels.dp.cpp
    solver/gmres_kernels.dp.cpp
    solver/cb_gmres_kernels.dp.cpp
    solver/idr_kernels.dp.cpp
    solver/ir_kernels.dp.cpp
    solver/lower_trs_kernels.dp.cpp
    solver/upper_trs_kernels.dp.cpp
    stop/criterion_kernels.dp.cpp
    stop/residual_norm_kernels.dp.cpp)

ginkgo_compile_features(ginkgo_dpcpp)

set(GINKGO_DPCPP_FLAGS ${GINKGO_COMPILER_FLAGS} -fsycl -fsycl-device-lib=libc)
set(GINKGO_DPCPP_FLAGS ${GINKGO_DPCPP_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_DPCPP_FLAGS}")
target_compile_features(ginkgo_dpcpp PRIVATE cxx_std_17)
target_link_libraries(ginkgo_dpcpp PUBLIC ginkgo_core_device)

ginkgo_default_includes(ginkgo_dpcpp)
ginkgo_install_library(ginkgo_dpcpp dpcpp)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_dpcpp)
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
