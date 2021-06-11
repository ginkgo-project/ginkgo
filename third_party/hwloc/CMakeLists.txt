message(STATUS "Configuring and building HWLOC")
set(TPL_HWLOC_PATH "${PROJECT_BINARY_DIR}/third_party/hwloc")
ginkgo_load_and_configure_package(hwloc_external "https://download.open-mpi.org/release/hwloc/v2.3/hwloc-2.3.0.tar.gz"
    "SHA1=69d8e2f088d365cf5e86d04f049c6b1e09e2b9c3"
    "${TPL_HWLOC_PATH}/src/configure" "--disable-nvml" "--disable-cuda" "--disable-rsmi"
    )

add_library(hwloc SHARED IMPORTED GLOBAL)
add_dependencies(hwloc hwloc_external )
file(MAKE_DIRECTORY ${TPL_HWLOC_PATH}/lib/)
file(GLOB HWLOC_LIBS "${TPL_HWLOC_PATH}/build/hwloc/.libs/libhwloc.so*")
configure_file("${TPL_HWLOC_PATH}/build/include/hwloc/autogen/config.h" "${TPL_HWLOC_PATH}/src/include/hwloc/autogen/config.h" COPYONLY)
foreach(lib ${HWLOC_LIBS})
    get_filename_component(lib_name ${lib} NAME)
    configure_file("${lib}" "${TPL_HWLOC_PATH}/lib/${lib_name}" COPYONLY)
endforeach()
set(HWLOC_LIBRARIES "${TPL_HWLOC_PATH}/lib/libhwloc.so" CACHE FILEPATH "The path to HWLOC library libhwloc.so" FORCE)
set(HWLOC_INCLUDE_DIRS "${TPL_HWLOC_PATH}/src/include" CACHE PATH "The directory containing the hwloc header, hwloc.h" FORCE)
set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION ${HWLOC_LIBRARIES})
set_target_properties(hwloc PROPERTIES INTERFACE_LINK_LIBRARIES ${HWLOC_LIBRARIES})
set_target_properties(hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIRS}")
