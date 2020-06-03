cmake_minimum_required(VERSION 3.9)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

if(MSVC)
    if(GINKGO_BUILD_SHARED_LIBS)
        ginkgo_switch_to_windows_dynamic("CXX")
        ginkgo_switch_to_windows_dynamic("C")
    else()
        ginkgo_switch_to_windows_static("CXX")
        ginkgo_switch_to_windows_static("C")
    endif()
endif()

include(CheckLanguage)
check_language(CUDA)

add_executable(test_install test_install.cpp)
target_compile_features(test_install PUBLIC cxx_std_11)
target_link_libraries(test_install PRIVATE Ginkgo::ginkgo)

if(GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    if(MSVC)
        if(GINKGO_BUILD_SHARED_LIBS)
            ginkgo_switch_to_windows_dynamic("CUDA")
        else()
            ginkgo_switch_to_windows_static("CUDA")
        endif()
    endif()
    add_executable(test_install_cuda test_install_cuda.cu)
    target_link_libraries(test_install_cuda PRIVATE Ginkgo::ginkgo)
endif()
