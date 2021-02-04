message(STATUS "Configuring and building HWLOC")
set(TPL_HWLOC_PATH "${PROJECT_BINARY_DIR}/third_party/hwloc")
if(MSVC OR WIN32)
    message(STATUS "Downloading HWLOC binaries for win32")
    ginkgo_download_file("https://download.open-mpi.org/release/hwloc/v2.3/hwloc-win32-build-2.3.0.zip"
        "${TPL_HWLOC_PATH}/hwloc.zip"
        "SHA1"
        "1d53dc8db809dd19d8281556f5f3785105aba97f"
        )
    file(ARCHIVE_EXTRACT INPUT "${TPL_HWLOC_PATH}/hwloc.zip"
        DESTINATION "${TPL_HWLOC_PATH}/hwloc"
        VERBOSE
        )
else()
    message(STATUS "Downloading HWLOC tars and building")
    ginkgo_load_and_configure_package(hwloc_external "https://download.open-mpi.org/release/hwloc/v2.3/hwloc-2.3.0.tar.gz"
        "SHA1=69d8e2f088d365cf5e86d04f049c6b1e09e2b9c3"
        "${TPL_HWLOC_PATH}/src/configure"
        )

endif()

if(MSVC OR WIN32)
    set(HWLOC_LIBRARIES "${TPL_HWLOC_PATH}/hwloc/hwloc-win32-build-2.3.0/lib/libhwloc.a" CACHE FILEPATH "The path to HWLOC library libhwloc.so" FORCE)
    set(HWLOC_INCLUDE_DIRS "${TPL_HWLOC_PATH}/hwloc/hwloc-win32-build-2.3.0/include" CACHE PATH "The directory containing the hwloc header, hwloc.h" FORCE)
else()
    add_library(hwloc SHARED IMPORTED GLOBAL)
    add_dependencies(hwloc hwloc_external )
    set(HWLOC_DIR "${TPL_HWLOC_PATH}" CACHE FILEPATH "The path to HWLOC")
    file(MAKE_DIRECTORY ${TPL_HWLOC_PATH}/lib/)
    file(GLOB HWLOC_LIBS "${TPL_HWLOC_PATH}/build/hwloc/.libs/libhwloc*")
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
endif()
