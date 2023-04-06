message(STATUS "Configuring and building HWLOC")
set(TPL_HWLOC_PATH "${PROJECT_BINARY_DIR}/third_party/hwloc/src")
ginkgo_load_and_configure_package(hwloc_external "https://download.open-mpi.org/release/hwloc/v2.4/hwloc-2.4.1.tar.gz"
    "SHA1=b94950e8958e1125ca75ecac0bc0259ee3d108c4"
    "${TPL_HWLOC_PATH}"
    "./configure" "--prefix=${TPL_HWLOC_PATH}/install"
        "--disable-nvml" "--disable-cuda" "--disable-rsmi"
    )

add_library(hwloc SHARED IMPORTED GLOBAL)
add_dependencies(hwloc hwloc_external)
# NOTE: if changing this (e.g. to `.a`), please update the special case in
# `cmake/information_helpers.cmake`
set(HWLOC_LIBRARIES "${TPL_HWLOC_PATH}/install/lib/libhwloc.so"
    CACHE FILEPATH "The path to HWLOC library libhwloc.so" FORCE)
set(HWLOC_INCLUDE_DIRS "${TPL_HWLOC_PATH}/install/include"
    CACHE PATH "The directory containing the hwloc header, hwloc.h" FORCE)
set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION ${HWLOC_LIBRARIES})
set_target_properties(hwloc PROPERTIES INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARIES})
set_target_properties(hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIRS}")
