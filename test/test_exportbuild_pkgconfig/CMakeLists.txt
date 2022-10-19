cmake_minimum_required(VERSION 3.9)
project(GinkgoExportBuildWithPkgConfigTest LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GINKGO REQUIRED IMPORTED_TARGET ginkgo)

message(STATUS "GINKGO_LIBRARIES ${GINKGO_LIBRARIES}")
message(STATUS "GINKGO_INCLUDE_DIRS ${GINKGO_INCLUDE_DIRS}")
message(STATUS "GINKGO_CFLAGS ${GINKGO_CFLAGS}")
message(STATUS "GINKGO_LDFLAGS ${GINKGO_LDFLAGS}")


# Here, we use test install without any data. We instantiate the
# interface only.
add_executable(test_exportbuild_pkgconfig ../test_install/test_install.cpp)
target_compile_features(test_exportbuild_pkgconfig PUBLIC cxx_std_14)
get_target_property(LINK_LIBRARIES PkgConfig::GINKGO INTERFACE_LINK_LIBRARIES)
message(STATUS "LINK_LIBRARIES ${LINK_LIBRARIES}")
get_target_property(INC PkgConfig::GINKGO INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "INC ${INC}")
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE PkgConfig::GINKGO)
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE -l/usr/lib/libpapi.so)
# CMake PkgConfig only puts the -l, -L, and -framework into link_libraries and others into link_options
# When linking the target, the linking option will be before the compiled object such that the linking error
set_property(TARGET PkgConfig::GINKGO PROPERTY INTERFACE_LINK_LIBRARIES "${GINKGO_LDFLAGS}")
set_property(TARGET PkgConfig::GINKGO PROPERTY INTERFACE_LINK_OPTIONS "")
target_link_libraries(test_exportbuild_pkgconfig PRIVATE PkgConfig::GINKGO)
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_LIBRARIES})
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_LDFLAGS})
# target_include_directories(test_exportbuild_pkgconfig PUBLIC ${GINKGO_INCLUDE_DIRS})
