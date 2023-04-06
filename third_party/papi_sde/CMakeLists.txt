message(STATUS "Configuring and building PAPI-SDE")
set(TPL_PAPI_PATH "${PROJECT_BINARY_DIR}/third_party/papi_sde/src")
ginkgo_load_and_configure_package(papi_external
    "https://bitbucket.org/terry_cojean/papi/get/77cdd0ba8db98d86c1459dd5f55013aba242d5d5.tar.gz"
    "SHA1=540c18a14eeafb83cd60cbbf0a96706111dbff3b"
    "${TPL_PAPI_PATH}/src"
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
