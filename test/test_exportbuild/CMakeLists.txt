cmake_minimum_required(VERSION 3.16)
project(GinkgoExportBuildTest LANGUAGES CXX)

find_package(Ginkgo REQUIRED
             ONLY_CMAKE_FIND_ROOT_PATH)

# Here, we use test install without any data. We instantiate the
# interface only.
add_executable(test_exportbuild ../test_install/test_install.cpp)
target_compile_features(test_exportbuild PUBLIC cxx_std_14)
target_link_libraries(test_exportbuild PRIVATE Ginkgo::ginkgo)
