cmake_minimum_required(VERSION 3.9)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
    PATHS # The Path where ginkgo was installed
    # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
    )

# Copy the data to the build directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)

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

set(HAS_REFERENCE 0)
if (GINKGO_BUILD_REFERENCE)
    set(HAS_REFERENCE 1)
endif()
add_executable(test_install test_install.cpp)
target_compile_features(test_install PUBLIC cxx_std_14)
target_compile_definitions(test_install PRIVATE HAS_REFERENCE=${HAS_REFERENCE})
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
    set_source_files_properties(test_install.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(test_install_cuda test_install.cpp)
    target_compile_definitions(test_install_cuda PRIVATE HAS_CUDA=1)
    target_link_libraries(test_install_cuda PRIVATE Ginkgo::ginkgo)
endif()

if(GINKGO_BUILD_HIP)
    find_package(HIP REQUIRED)

    set_source_files_properties(test_install.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    # The library was compiled with `-fPIC` when in shared library mode
    if (GINKGO_BUILD_SHARED_LIBS)
        set (GINKGO_PIC_OPTION "-fPIC")
        set (GINKGO_CUDA_PIC_OPTION "-Xcompiler '-fPIC'")
    endif()
    hip_add_executable(test_install_hip test_install.cpp
        HIPCC_OPTIONS "-std=c++14"
        CLANG_OPTIONS "${GINKGO_PIC_OPTION}"
        NVCC_OPTIONS "${GINKGO_CUDA_PIC_OPTION}")

    target_link_libraries(test_install_hip PRIVATE Ginkgo::ginkgo)
    target_compile_definitions(test_install_hip PRIVATE HAS_HIP=1)

    # If we always link with CXX there is no RPATH issue
    set_target_properties(test_install_hip PROPERTIES LINKER_LANGUAGE CXX)

    # Instead of using CXX, it's possible to instead use
    # target_link_libraries(test_install_hip PRIVATE ${GINKGO_INSTALL_RPATH_FOR_HIP})
endif()
