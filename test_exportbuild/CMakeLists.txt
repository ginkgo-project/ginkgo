cmake_minimum_required(VERSION 3.9)
project(GinkgoExportBuildTest LANGUAGES CXX)

find_package(Ginkgo REQUIRED)

if(MSVC)
    if(GINKGO_BUILD_SHARED_LIBS)
        ginkgo_switch_to_windows_dynamic("CXX")
        ginkgo_switch_to_windows_dynamic("C")
    else()
        ginkgo_switch_to_windows_static("CXX")
        ginkgo_switch_to_windows_static("C")
    endif()
endif()

add_executable(test_exportbuild ../test_install/test_install.cpp)
target_compile_features(test_exportbuild PUBLIC cxx_std_14)
target_link_libraries(test_exportbuild PRIVATE Ginkgo::ginkgo)
