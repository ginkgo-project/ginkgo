message(STATUS "Configuring and building HWLOC")
if(WIN32)
  ginkgo_download_file("https://download.open-mpi.org/release/hwloc/v2.3/hwloc-win32-build-2.3.0.zip"
    "${PROJECT_BINARY_DIR}/third_party/hwloc/hwloc.zip"
    "SHA1"
    "1d53dc8db809dd19d8281556f5f3785105aba97f"
    )
  file(ARCHIVE_EXTRACT INPUT "${PROJECT_BINARY_DIR}/third_party/hwloc/hwloc.zip")
elseif(WIN64)
  ginkgo_download_file("https://download.open-mpi.org/release/hwloc/v2.3/hwloc-win64-build-2.3.0.zip"
    "${PROJECT_BINARY_DIR}/third_party/hwloc/hwloc.zip"
    "SHA1"
    "943e0ca50262dcf256b5956c73faded0141e06b9"
    )
  file(ARCHIVE_EXTRACT INPUT "${PROJECT_BINARY_DIR}/third_party/hwloc/hwloc.zip")
else()
ginkgo_load_and_configure_package(hwloc_external "https://download.open-mpi.org/release/hwloc/v2.3/hwloc-2.3.0.tar.gz"
    "SHA1=69d8e2f088d365cf5e86d04f049c6b1e09e2b9c3"
    "${CMAKE_BINARY_DIR}/third_party/hwloc/src/configure"
    )

endif()

if(WIN32 OR WIN64)
  add_library(hwloc STATIC IMPORTED GLOBAL)
  add_dependencies(hwloc hwloc_external)
  set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third_party/hwloc/hwloc/lib/libhwloc.a" )
  set_target_properties(hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/third_party/hwloc/hwloc/include")
else()
  add_library(hwloc SHARED IMPORTED GLOBAL)
  add_dependencies(hwloc hwloc_external )
  set_target_properties(hwloc PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/third_party/hwloc/build/hwloc/.libs/libhwloc.so" )
  set_target_properties(hwloc PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_BINARY_DIR}/third_party/hwloc/build/include")
endif()
