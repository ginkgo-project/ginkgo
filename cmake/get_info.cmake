SET(detailed_log "${CMAKE_BINARY_DIR}/detailed.log")
SET(minimal_log  "${CMAKE_BINARY_DIR}/minimal.log")
FILE(REMOVE ${detailed_log} ${minimal_log})

MACRO(_both)
  # Write to both log files:
  FILE(APPEND ${detailed_log} "${ARGN}")
  FILE(APPEND ${minimal_log} "${ARGN}")
ENDMACRO()

MACRO(_detailed)
  # Only write to detailed.log:
  FILE(APPEND ${detailed_log} "${ARGN}")
ENDMACRO()

MACRO(_minimal)
  # Only write to minimal.log:
  FILE(APPEND ${minimal_log} "${ARGN}")
ENDMACRO()

FUNCTION(build_type_spec log_type var_name)
    if(CMAKE_BUILD_TYPE MATCHES "Release")
        set(var_string "${var_name}_RELEASE")
        set(upd_string "\n#        ${var_name}_RELEASE                   ${${var_string}} ")
        FILE(APPEND ${log_type} ${upd_string})
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "Debug" )
        set(var_string "${var_name}_DEBUG")
        set(upd_string "#       ${var_name}_DEBUG                 ${${var_string}} ")
        FILE(APPEND ${log_type} "${upd_string}")
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "RelWithDebInfo")
        set(var_string "${var_name}_RELWITHDEBINFO")
        set(upd_string "#       ${var_name}_RELWITHDEBINFO                      ${${var_string}} ")
        FILE(APPEND ${log_type} "${upd_string}")
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "MinSizeRel")
        set(var_string "${var_name}_MINSIZEREL")
        set(upd_string "#       ${var_name}_MINSIZEREL                  ${${var_string}} ")
        FILE(APPEND ${log_type} "${upd_string}")
    endif()
ENDFUNCTION()

_both(
"


"
)

IF("${GINKGO_GIT_SHORTREV}" STREQUAL "")
    _both("
#########################################################################
#
#  Summary of Configuration for  (Ginkgo version ${Ginkgo_VERSION_TAG})\n")
ELSE()
    _both("
#########################################################################
#
#  Summary of Configuration for  (Ginkgo version ${Ginkgo_VERSION_TAG}, shortrev ${GINKGO_GIT_SHORTREV})\n")
ENDIF()
_both(
"#
#
#  Ginkgo configuration:
#        CMAKE_BUILD_TYPE:                       ${CMAKE_BUILD_TYPE}
#        BUILD_SHARED_LIBS:                      ${BUILD_SHARED_LIBS}
#        CMAKE_INSTALL_PREFIX:                   ${CMAKE_INSTALL_PREFIX}
#        CMAKE_SOURCE_DIR:                       ${CMAKE_SOURCE_DIR}
#
#  User Configuration options:
#
#      Enabled modules:
#        GINKGO_BUILD_OMP:                       ${GINKGO_BUILD_OMP}
#        GINKGO_BUILD_REFERENCE:                 ${GINKGO_BUILD_REFERENCE}
#        GINKGO_BUILD_CUDA:                      ${GINKGO_BUILD_CUDA}
#        GINKGO_BUILD_HIP:                       ${GINKGO_BUILD_HIP}
#
#     Tests, benchmarks and examples:
#        GINKGO_BUILD_TESTS:                     ${GINKGO_DEVEL_TOOLS}
#        GINKGO_BUILD_EXAMPLES:                  ${GINKGO_BUILD_EXAMPLES}
#        GINKGO_EXTLIB_EXAMPLE:                  ${GINKGO_EXTLIB_EXAMPLE}
#        GINKGO_BUILD_BENCHMARKS:                ${GINKGO_BUILD_BENCHMARKS}
#
#     Documentation:
#        GINKGO_BUILD_DOC:                       ${GINKGO_BUILD_DOC}
#        GINKGO_VERBOSE_LEVEL:                   ${GINKGO_VERBOSE_LEVEL}
#
#     Developer helpers:
#        GINKGO_DEVEL_TOOLS:                     ${GINKGO_DEVEL_TOOLS}
#        GINKGO_WITH_CLANG_TIDY:                 ${GINKGO_WITH_CLANG_TIDY}
#        GINKGO_WITH_IWYU:                       ${GINKGO_WITH_IWYU}
#        GINKGO_CHECK_CIRCULAR_DEPS:             ${GINKGO_CHECK_CIRCULAR_DEPS}
"
  )
_both(
"#
#     General information:
#        CMAKE_BINARY_DIR:                       ${CMAKE_BINARY_DIR}
#        CMAKE_SOURCE_DIR:                       ${CMAKE_SOURCE_DIR}
#        CMAKE_CXX_COMPILER:                     ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on platform ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}
#                                                ${CMAKE_CXX_COMPILER}"
  )



_detailed(
    "
#
#########################################################################
#
#   Compiled Modules
#"
    )
_detailed(
    "
#########################################################################
#
#    The Core module is being compiled.
#
#        BUILD_SHARED_LIBS:                      ${BUILD_SHARED_LIBS}"
    )
IF(CMAKE_C_COMPILER_WORKS)
    _detailed(
        "
#        CMAKE_C_COMPILER:                       ${CMAKE_C_COMPILER}"
        )
    build_type_spec(${detailed_log} "CMAKE_C_FLAGS")
ENDIF()
IF(CMAKE_CXX_COMPILER_WORKS)
    _detailed(
        "
#        CMAKE_CXX_COMPILER:                     ${CMAKE_CXX_COMPILER}"
        )
    build_type_spec(${detailed_log} "CMAKE_CXX_FLAGS")
ENDIF()
IF(CMAKE_Fortran_COMPILER_WORKS)
    _detailed(
        "
#        CMAKE_Fortran_COMPILER:                 ${CMAKE_Fortran_COMPILER}"
        )
ENDIF()
_detailed(
    "
#        CMAKE_GENERATOR:                        ${CMAKE_GENERATOR}
#"
    )

IF(GINKGO_BUILD_REFERENCE)
    _detailed(
        "
#########################################################################
#
#    The Reference module is being compiled.
#
#        GINKGO_BUILD_REFERENCE:                 ${GINKGO_BUILD_REFERENCE}
#        GINKGO_COMPILER_FLAGS:                  ${GINKGO_COMPILER_FLAGS}
#
#"
        )
ENDIF()


IF(GINKGO_BUILD_OMP)
    _detailed(
        "
#########################################################################
#
#    The OMP module is being compiled.
#
#    CMake related OMP variables:
#        OpenMP_CXX_FLAGS:                        ${OpenMP_CXX_FLAGS}"
        )
    _detailed(
"
#        OpenMP_CXX_LIB_NAMES:                    ${OpenMP_CXX_LIB_NAMES}
#        OpenMP_CXX_LIBRARIES:                    ${OpenMP_CXX_LIBRARIES}
#
#    Ginkgo specific OMP variables:
#        GINKGO_COMPILER_FLAGS:                   ${GINKGO_COMPILER_FLAGS}
#
#"
        )
ENDIF()


IF(GINKGO_BUILD_CUDA)
    _detailed(
        "
#########################################################################
#
#    The CUDA module is being compiled.
#
#    CMake related CUDA variables:
#        CMAKE_CUDA_COMPILER:                    ${CMAKE_CUDA_COMPILER}
#        CMAKE_CUDA_COMPILER_VERSION:            ${CMAKE_CUDA_COMPILER_VERSION}"
        )
    build_type_spec(${detailed_log} "CMAKE_CUDA_FLAGS")
    _detailed(
        "
#        CMAKE_CUDA_HOST_COMPILER:               ${GINKGO_CUDA_HOST_COMPILER}
#        CUDA_INCLUDE_DIRS:                      ${CUDA_INCLUDE_DIRS}
#
#    Ginkgo specific CUDA variables:
#        GINKGO_CUDA_ARCHITECTURES:              ${GINKGO_CUDA_ARCHITECTURES}
#        GINKGO_CUDA_COMPILER_FLAGS:             ${GINKGO_CUDA_COMPILER_FLAGS}
#        GINKGO_CUDA_DEFAULT_HOST_COMPILER:      ${GINKGO_CUDA_DEFAULT_HOST_COMPILER}
#
#    CUDA libraries:
#        CUBLAS:                                 ${CUBLAS}
#        CUDA_RUNTIME_LIBS:                      ${CUDA_RUNTIME_LIBS}
#        CUSPARSE:                               ${CUSPARSE}
#"
        )
ENDIF()

IF(GINKGO_BUILD_HIP)
    _detailed(
        "
#########################################################################
#
#    The HIP module is being compiled.
#
#    Ginkgo specific HIP variables:
#        GINKGO_HIPCONFIG_PATH:                 ${GINKGO_HIPCONFIG_PATH}
#        GINKGO_HIP_AMDGPU:                     ${GINKGO_HIP_AMDGPU}
#        GINKGO_HIP_CLANG_COMPILER_FLAGS:       ${GINKGO_HIP_CLANG_COMPILER_FLAGS}
#        GINKGO_HIP_HCC_COMPILER_FLAGS:         ${GINKGO_HCC_COMPILER_FLAGS}
#        GINKGO_HIP_NVCC_COMPILER_FLAGS:        ${GINKGO_HIP_NVCC_COMPILER_FLAGS}
#        GINKGO_HIP_THRUST_PATH:                ${GINKGO_HIP_THRUST_PATH}
#        GINKGO_HIPCC_OPTIONS:                  ${GINKGO_HIPCC_OPTIONS}
#        GINKGO_HIP_NVCC_OPTIONS:               ${GINKGO_HIP_NVCC_OPTIONS}
#        GINKGO_HIP_HCC_OPTIONS:                ${GINKGO_HIP_HCC_OPTIONS}
#        GINKGO_HIP_CLANG_OPTIONS:              ${GINKGO_HIP_CLANG_OPTIONS}
#
#    HIP variables:
#        HIP_VERSION:                           ${HIP_VERSION}
#        HIP_COMPILER:                          ${HIP_COMPILER}
#        HIP_PATH:                              ${HIP_PATH}
#        ROCM_PATH:                             ${ROCM_PATH}
#        HIP_PLATFORM:                          ${HIP_PLATFORM}
#        HIP_ROOT_DIR:                          ${HIP_ROOT_DIR}
#        HCC_PATH:                              ${HCC_PATH}
#        HIP_RUNTIME:                           ${HIP_RUNTIME}
#        HIPBLAS_PATH:                          ${HIPBLAS_PATH}
#        HIPSPARSE_PATH:                        ${HISPARSE_PATH}
#        HIP_CLANG_INCLUDE_PATH:                ${HIP_CLANG_INCLUDE_PATH}"
        )
    build_type_spec(${detailed_log} "HIP_CLANG_FLAGS")
    _detailed(
        "
#        HIP_CLANG_PATH:                        ${HIP_CLANG_PATH}"
        )
    build_type_spec(${detailed_log} "HIP_HCC_FLAGS")
    _detailed(
        "
#        HIP_HIPCC_CMAKE_LINKER_HELPER:         ${HIP_CLANG_INCLUDE_PATH}"
        )
    build_type_spec(${detailed_log} "HIP_HIPCC_FLAGS")
    _detailed(
        "
#        HIP_HIPCC_EXECUTABLE:                  ${HIP_CLANG_INCLUDE_PATH}
#        HIP_HIPCONFIG_EXECUTABLE:              ${HIP_HIPCONFIG_EXECUTABLE}
#        HIP_HOSTCOMPILATION_CPP:               ${HIP_HOSTCOMPILATION_CPP}"
        )
    build_type_spec(${detailed_log} "HIP_NVCC_FLAGS")
    _detailed(
        "
#
#"
        )
ENDIF()

    _detailed(
        "
#########################################################################
#
#    Optional Components
#
#        GKO_HAVE_PAPI_SDE:                     ${GKO_HAVE_PAPI_SDE}"
        )
    if(PAPI_sde_FOUND)
    _detailed(
        "
#
#        PAPI_VERSION                           ${PAPI_VERSION}
#        PAPI_INCLUDE_DIR                       ${PAPI_INCLUDE_DIR}
#        PAPI_LIBRARY_RELEASE                   ${PAPI_LIBRARY_RELEASE}
#\n"
        )
    endif()
    _detailed(
        "
#\n"
        )


_minimal(
"
#\n#  Detailed information (More compiler flags, module configuration) can be found in detailed.log
#\n# Now, run "
  )
IF(CMAKE_GENERATOR MATCHES "Ninja")
  _minimal("ninja")
ELSE()
_minimal("make")
ENDIF()
_minimal(" to compile Ginkgo!\n")
_both("#
#########################################################################\n")
