message(STATUS "Configuring and building PAPI-SDE")
set(TPL_PAPI_PATH "${PROJECT_BINARY_DIR}/third_party/papi_sde/src")
ginkgo_load_and_configure_package(papi_external
    "https://github.com/icl-utk-edu/papi/archive/d2dd17a07a3c175fbb26ce5528671e3a7e00b80f.tar.gz"
    "SHA1=55019037c47aff216ff831b4191e7147f6932464"
    "${TPL_PAPI_PATH}/build"
    "./configure" "--prefix=${TPL_PAPI_PATH}/install"
        "--with-components=sde" "--with-libsde=yes" "--with-tests=no"
        "--with-static-lib=no" "--with-shared-lib=yes"
    )

add_library(PAPI SHARED IMPORTED GLOBAL)
add_library(PAPI::PAPI ALIAS PAPI)
add_dependencies(PAPI papi_external)
# NOTE: if changing this (e.g. to `.a`), please update the special case in
# `cmake/information_helpers.cmake`
set(PAPI_LIBRARIES "${TPL_PAPI_PATH}/install/lib/libpapi.so"
    CACHE FILEPATH "The path to PAPI libraries" FORCE)
set(PAPI_INCLUDE_DIRS "${TPL_PAPI_PATH}/install/include" CACHE PATH
    "The directory containing the PAPI header, papi.h" FORCE)
set_target_properties(PAPI PROPERTIES IMPORTED_LOCATION "${PAPI_LIBRARIES}")
set_target_properties(PAPI PROPERTIES INTERFACE_LINK_LIBRARIES "${PAPI_LIBRARIES}")
set_target_properties(PAPI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INCLUDE_DIRS}")


add_library(PAPI_SDE SHARED IMPORTED GLOBAL)
add_library(PAPI::PAPI_SDE ALIAS PAPI_SDE)
add_dependencies(PAPI_SDE papi_external)
# NOTE: if changing this (e.g. to `.a`), please update the special case in
# `cmake/information_helpers.cmake`
set(PAPI_SDE_LIBRARIES "${TPL_PAPI_PATH}/install/lib/libsde.so"
    CACHE FILEPATH "The path to PAPI SDE libraries" FORCE)
set(PAPI_SDE_INCLUDE_DIRS "${TPL_PAPI_PATH}/install/include" CACHE PATH
    "The directory containing the PAPI SDE header, sde_lib.h" FORCE)
set_target_properties(PAPI_SDE PROPERTIES IMPORTED_LOCATION "${PAPI_SDE_LIBRARIES}")
set_target_properties(PAPI_SDE PROPERTIES INTERFACE_LINK_LIBRARIES "${PAPI_SDE_LIBRARIES}")
set_target_properties(PAPI_SDE PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PAPI_SDE_INCLUDE_DIRS}")
