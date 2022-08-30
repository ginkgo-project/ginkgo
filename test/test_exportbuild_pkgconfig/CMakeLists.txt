cmake_minimum_required(VERSION 3.9)
project(GinkgoExportBuildWithPkgConfigTest LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
# find_package(Ginkgo REQUIRED)
pkg_check_modules(GINKGO REQUIRED ginkgo)

# Here, we use test install without any data. We instantiate the
# interface only.
add_executable(test_exportbuild_pkgconfig ../test_install/test_install.cpp)
target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_LIBRARIES})
target_include_directories(test_exportbuild_pkgconfig PRIVATE ${GINKGO_INCLUDE_DIRS})
target_compile_features(test_exportbuild_pkgconfig PUBLIC cxx_std_14)
target_link_libraries(test_exportbuild_pkgconfig PRIVATE ${GINKGO_CFLAGS})