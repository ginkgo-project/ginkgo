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
target_compile_features(test_install PUBLIC cxx_std_14)
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

if(GINKGO_BUILD_HIP
   AND GINKGO_HIP_PLATFORM MATCHES "hcc"
   AND GINKGO_HIP_VERSION VERSION_GREATER_EQUAL "3.5"
   AND NOT GINKGO_BUILD_SHARED_LIBS)
    # Compile options somehow add hip-clang specific flags. Wipe them.
    # Currently, the flags wiped out should be:
    # -x;hip;--hip-device-lib-path=/opt/rocm/lib;--cuda-gpu-arch=gfx900;
    # --cuda-gpu-arch=gfx906
    set_target_properties(hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "")
    # In addition, link libraries have a similar problem. We only keep
    # `hip::host`. Currently, the flags should be:
    # hip::host;--hip-device-lib-path=/opt/rocm/lib;--hip-link;
    # --cuda-gpu-arch=gfx900;--cuda-gpu-arch=gfx906
    set_target_properties(hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host")
endif()
