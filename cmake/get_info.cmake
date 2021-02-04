SET(detailed_log "${PROJECT_BINARY_DIR}/detailed.log")
SET(minimal_log  "${PROJECT_BINARY_DIR}/minimal.log")
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

function(ginkgo_print_generic_header log_type optional_string)
    set(upd_string
        "
---------------------------------------------------------------------------------------------------------
--
--    ${optional_string}"
        )
    FILE(APPEND ${log_type} "${upd_string}")
endfunction()

function(ginkgo_print_module_header log_type module_name)
    set(upd_string
        "The ${module_name} module is being compiled.
--
--    CMake related ${module_name} module variables:"
        )
    ginkgo_print_generic_header(${log_type} "${upd_string}")
endfunction()

function(ginkgo_print_module_footer log_type optional_string)
    set(upd_string
        "
--    ${optional_string}"
        )
    FILE(APPEND ${log_type} "${upd_string}")
endfunction()

FUNCTION(ginkgo_print_flags log_type var_name)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" suff)
    set(var_string "${var_name}_${suff}")
    if(${var_string} STREQUAL "")
        set(str_value "<empty>")
    else()
        set(str_value "${${var_string}}")
    endif()
    string(SUBSTRING "
--        ${var_string}:                                                        " 0 55 upd_string)
    string(APPEND upd_string "${str_value}")
    FILE(APPEND ${log_type} ${upd_string})
ENDFUNCTION()

function(ginkgo_print_variable log_type var_name)
    string(SUBSTRING
        "
--        ${var_name}:                                                          " 0 55 upd_string)
    if(${var_name} STREQUAL "")
        set(str_value "<empty>")
    else()
        set(str_value "${${var_name}}")
    endif()
    string(APPEND upd_string "${str_value}")
    FILE(APPEND ${log_type} "${upd_string}")
endfunction()


function(ginkgo_print_env_variable log_type var_name)
    string(SUBSTRING
        "
--        ${var_name}:                                                          " 0 55 upd_string)
    if(NOT ENV{${var_name}})
        set(str_value "<empty>")
    else()
        set(str_value "$ENV{${var_name}}")
    endif()
    string(APPEND upd_string "${str_value}")
    FILE(APPEND ${log_type} "${upd_string}")
endfunction()


macro(ginkgo_print_foreach_variable log_type)
    foreach(var ${ARGN})
        ginkgo_print_variable(${log_type} ${var} )
    endforeach()
endmacro()

IF("${GINKGO_GIT_SHORTREV}" STREQUAL "")
    set(to_print "Summary of Configuration for Ginkgo (version ${Ginkgo_VERSION} with tag ${Ginkgo_VERSION_TAG})
--"
        )
    ginkgo_print_generic_header(${detailed_log} "${to_print}")
    ginkgo_print_generic_header(${minimal_log} "${to_print}")
ELSE()
    set(to_print "Summary of Configuration for (Ginkgo version ${Ginkgo_VERSION} with tag ${Ginkgo_VERSION_TAG}, shortrev ${GINKGO_GIT_SHORTREV})"
        )
    ginkgo_print_generic_header(${detailed_log} "${to_print}")
    ginkgo_print_generic_header(${minimal_log} "${to_print}")
ENDIF()

set(log_types "detailed_log;minimal_log")
foreach(log_type ${log_types})
    ginkgo_print_module_footer(${${log_type}} "Ginkgo configuration:")
    ginkgo_print_foreach_variable(${${log_type}}
        "CMAKE_BUILD_TYPE;BUILD_SHARED_LIBS;CMAKE_INSTALL_PREFIX"
        "PROJECT_SOURCE_DIR;PROJECT_BINARY_DIR")
    string(SUBSTRING
        "
--        CMAKE_CXX_COMPILER:                                                   " 0 55 print_string)
    set(str2 "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} on platform ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
    string(APPEND print_string "${str2}")
    FILE(APPEND ${${log_type}} "${print_string}")
    string(SUBSTRING
        "
--                                                                              " 0 55 print_string)
    set(str2 "${CMAKE_CXX_COMPILER}")
    string(APPEND print_string "${str2}")
    FILE(APPEND ${${log_type}} "${print_string}")
    ginkgo_print_module_footer(${${log_type}} "User configuration:")
    ginkgo_print_module_footer(${${log_type}} "  Enabled modules:")
    ginkgo_print_foreach_variable(${${log_type}}
        "GINKGO_BUILD_OMP;GINKGO_BUILD_REFERENCE;GINKGO_BUILD_CUDA;GINKGO_BUILD_HIP;GINKGO_BUILD_DPCPP")
    ginkgo_print_module_footer(${${log_type}} "  Tests, benchmarks and examples:")
    ginkgo_print_foreach_variable(${${log_type}}
        "GINKGO_BUILD_TESTS;GINKGO_BUILD_EXAMPLES;GINKGO_EXTLIB_EXAMPLE;GINKGO_BUILD_BENCHMARKS;GINKGO_BENCHMARK_ENABLE_TUNING")
    ginkgo_print_module_footer(${${log_type}} "  Documentation:")
    ginkgo_print_foreach_variable(${${log_type}} "GINKGO_BUILD_DOC;GINKGO_VERBOSE_LEVEL")
    ginkgo_print_module_footer(${${log_type}} "  Developer helpers:")
    ginkgo_print_foreach_variable(${${log_type}}
        "GINKGO_DEVEL_TOOLS;GINKGO_WITH_CLANG_TIDY;GINKGO_WITH_IWYU"
        "GINKGO_CHECK_CIRCULAR_DEPS;GINKGO_CHECK_PATH")
    ginkgo_print_module_footer(${${log_type}} "")
endforeach()

set(to_print
    "Compiled Modules
--"
    )
ginkgo_print_generic_header(${detailed_log} "${to_print}")

include(core/get_info.cmake)

IF(GINKGO_BUILD_REFERENCE)
    include(reference/get_info.cmake)
ENDIF()

IF(GINKGO_BUILD_OMP)
    include(omp/get_info.cmake)
ENDIF()

IF(GINKGO_BUILD_CUDA)
    include(cuda/get_info.cmake)
ENDIF()

IF(GINKGO_BUILD_HIP)
    include(hip/get_info.cmake)
ENDIF()

IF(GINKGO_BUILD_DPCPP)
    include(dpcpp/get_info.cmake)
ENDIF()

ginkgo_print_generic_header(${minimal_log} "  Components:")
ginkgo_print_generic_header(${detailed_log} "  Components:")
if(PAPI_sde_FOUND)
    ginkgo_print_variable(${detailed_log} "PAPI_VERSION")
    ginkgo_print_variable(${detailed_log} "PAPI_INCLUDE_DIR")
    ginkgo_print_flags(${detailed_log} "PAPI_LIBRARY")
endif()
ginkgo_print_variable(${minimal_log} "GINKGO_BUILD_HWLOC")
ginkgo_print_variable(${detailed_log} "GINKGO_BUILD_HWLOC")
ginkgo_print_variable(${detailed_log} "HWLOC_VERSION")
ginkgo_print_variable(${detailed_log} "GKO_HWLOC_XMLFILE")
ginkgo_print_variable(${detailed_log} "HWLOC_LIBRARIES")
ginkgo_print_variable(${detailed_log} "HWLOC_INCLUDE_DIRS")

_minimal(
    "
--\n--  Detailed information (More compiler flags, module configuration) can be found in detailed.log
--   ")
_both(
    "\n--\n--  Now, run  cmake --build .  to compile Ginkgo!\n"
    )
_both("--
---------------------------------------------------------------------------------------------------------\n")
