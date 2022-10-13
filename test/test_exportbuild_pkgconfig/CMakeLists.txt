cmake_minimum_required(VERSION 3.9)
project(GinkgoExportBuildWithPkgConfigTest LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GINKGO REQUIRED IMPORTED_TARGET ginkgo)

message("GINKGO_LIBRARIES ${GINKGO_LIBRARIES}")
message("GINKGO_INCLUDE_DIRS ${GINKGO_INCLUDE_DIRS}")
message("GINKGO_CFLAGS ${GINKGO_CFLAGS}")
message("GINKGO_LDFLAGS ${GINKGO_LDFLAGS}")


# Here, we use test install without any data. We instantiate the
# interface only.
add_executable(test_exportbuild_pkgconfig ../test_install/test_install.cpp)
target_compile_features(test_exportbuild_pkgconfig PUBLIC cxx_std_14)
target_link_libraries(test_exportbuild_pkgconfig PUBLIC PkgConfig::GINKGO)
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_LIBRARIES})
# target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_LDFLAGS})
# target_include_directories(test_exportbuild_pkgconfig PRIVATE ${GINKGO_INCLUDE_DIRS})
# target_compile_features(test_exportbuild_pkgconfig PRIVATE ${GINKGO_CFLAGS})
