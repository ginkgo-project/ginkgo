cmake_minimum_required(VERSION 3.16)
project(GinkgoExportBuildWithPkgConfigTest LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GINKGO REQUIRED IMPORTED_TARGET ginkgo)


# Here, we use test install without any data. We instantiate the
# interface only.
add_executable(test_pkgconfig ../test_install/test_install.cpp)
target_compile_features(test_pkgconfig PUBLIC cxx_std_14)
# CMake PkgConfig only puts the -l, -L, and -framework into link_libraries and others into link_options
# When linking the target, the linking option will be before the compiled object to lead the linking error
set_property(TARGET PkgConfig::GINKGO PROPERTY INTERFACE_LINK_LIBRARIES "${GINKGO_LDFLAGS}")
set_property(TARGET PkgConfig::GINKGO PROPERTY INTERFACE_LINK_OPTIONS "")
target_link_libraries(test_pkgconfig PRIVATE PkgConfig::GINKGO)
