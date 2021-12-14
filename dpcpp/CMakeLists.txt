if (NOT GKO_CAN_COMPILE_DPCPP)
    message(FATAL_ERROR "The CMAKE_CXX_COMPILER compiler, which is "
        "${CMAKE_CXX_COMPILER} cannot compile DPC++ code!")
endif()

ginkgo_extract_dpcpp_version(${CMAKE_CXX_COMPILER} GINKGO_DPCPP_VERSION)
set(GINKGO_DPCPP_VERSION ${GINKGO_DPCPP_VERSION} PARENT_SCOPE)

find_package(MKL CONFIG REQUIRED HINTS "$ENV{MKLROOT}")
set(GINKGO_MKL_ROOT "${MKL_ROOT}" PARENT_SCOPE)
find_package(oneDPL REQUIRED HINTS "$ENV{DPL_ROOT}")
set(GINKGO_DPL_ROOT "${DPL_ROOT}" PARENT_SCOPE)

add_library(ginkgo_dpcpp $<TARGET_OBJECTS:ginkgo_dpcpp_device> "")
target_sources(ginkgo_dpcpp
    PRIVATE
    base/device_matrix_data_kernels.dp.cpp
    base/executor.dp.cpp
    base/helper.dp.cpp
    base/index_set_kernels.dp.cpp
    base/version.dp.cpp
    components/prefix_sum_kernels.dp.cpp
    distributed/partition_kernels.dp.cpp
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
    matrix/fft_kernels.dp.cpp
    matrix/sellp_kernels.dp.cpp
    matrix/sparsity_csr_kernels.dp.cpp
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
    ${GKO_UNIFIED_COMMON_SOURCES}
    )

## Parse the subgroup and workgroup lists
# 1. Sort from smallest element to the biggest
insertionSortRecursive(GINKGO_DPCPP_SUBGROUPS)
insertionSortRecursive(GINKGO_DPCPP_WORKGROUPS)
# 2. Extract the two biggest elements
list(GET GINKGO_DPCPP_SUBGROUPS -1 GINKGO_DPCPP_BIGGEST_SUBGROUP)
list(GET GINKGO_DPCPP_WORKGROUPS -1 GINKGO_DPCPP_BIGGEST_WORKGROUP)
# 3. Combine the two lists, also format the workgroup list
set(GINKGO_DPCPP_1D_CFG "")
set(GINKGO_DPCPP_WORKGROUPS_FORMATTED "")
foreach(workgroup ${GINKGO_DPCPP_WORKGROUPS})
    foreach(subgroup ${GINKGO_DPCPP_SUBGROUPS})
        if (GINKGO_DPCPP_1D_CFG STREQUAL "")
            set(GINKGO_DPCPP_1D_CFG "KCFG_1D::encode(${workgroup}, ${subgroup})")
        else()
            set(GINKGO_DPCPP_1D_CFG "${GINKGO_DPCPP_1D_CFG}, KCFG_1D::encode(${workgroup}, ${subgroup})")
        endif()
    endforeach()
    if (GINKGO_DPCPP_WORKGROUPS_FORMATTED STREQUAL "")
        set(GINKGO_DPCPP_WORKGROUPS_FORMATTED "BlockCfg::encode(${workgroup})")
    else()
        set(GINKGO_DPCPP_WORKGROUPS_FORMATTED "BlockCfg::encode(${workgroup}), ${GINKGO_DPCPP_WORKGROUPS_FORMATTED}, ")
    endif()
endforeach()
# 4. Reverse the subgroup list, replace ; to ,
list(REVERSE GINKGO_DPCPP_SUBGROUPS)
string(REPLACE ";" "," GINKGO_DPCPP_SUBGROUPS_FORMATTED "${GINKGO_DPCPP_SUBGROUPS}")

## Now we can generate the DPC++ Configure file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/base/config.hpp.in
    ${CMAKE_CURRENT_SOURCE_DIR}/base/config.hpp @ONLY)


ginkgo_compile_features(ginkgo_dpcpp)
target_compile_definitions(ginkgo_dpcpp PRIVATE GKO_COMPILING_DPCPP _ONEDPL_COMPILE_KERNEL=0)

set(GINKGO_DPCPP_FLAGS ${GINKGO_DPCPP_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_DPCPP_FLAGS}")
target_compile_options(ginkgo_dpcpp PRIVATE "${GINKGO_COMPILER_FLAGS}")
target_compile_features(ginkgo_dpcpp PRIVATE cxx_std_17)

# Note: add MKL as PRIVATE not PUBLIC (MKL example shows) to avoid propagating
# find_package(MKL) everywhere when linking ginkgo (see the MKL example
# https://software.intel.com/content/www/us/en/develop/documentation/onemkl-windows-developer-guide/top/getting-started/cmake-config-for-onemkl.html)
target_compile_options(ginkgo_dpcpp PRIVATE $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(ginkgo_dpcpp PRIVATE $<TARGET_PROPERTY:MKL::MKL_DPCPP,INTERFACE_INCLUDE_DIRECTORIES>)
target_include_directories(ginkgo_dpcpp PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/core>)

target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-lib=all)

if (NOT GINKGO_DPCPP_AOT_TARGETS STREQUAL "")
    string(REPLACE ";" "," GINKGO_DPCPP_AOT_TARGETS_FORMATTED "${GINKGO_DPCPP_AOT_TARGETS}")
    target_compile_options(ginkgo_dpcpp PRIVATE -fsycl-targets=${GINKGO_DPCPP_AOT_TARGETS_FORMATTED})
else()
    # When building ginkgo as a static library, we need to use dpcpp and per_source
    # link option when the program uses a dpcpp related function.
    if (BUILD_SHARED_LIBS)
        target_link_options(ginkgo_dpcpp PRIVATE -fsycl-device-code-split=per_source)
    else ()
        target_link_options(ginkgo_dpcpp PUBLIC -fsycl-device-code-split=per_source)
    endif()
endif()
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
