cmake_minimum_required(VERSION 3.9)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
    PATHS # The Path where ginkgo was installed
    # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
    )


# Manage RPATH. This needs to be before the targets.
set(CMAKE_SKIP_BUILD_RPATH OFF)
set(CMAKE_SKIP_INSTALL_RPATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

# Copy the data to the build directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)

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
if(GINKGO_BUILD_MPI)
    find_package(MPI 3.1 COMPONENTS CXX REQUIRED)
    target_link_libraries(test_install PRIVATE MPI::MPI_CXX)
endif()

if(GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    configure_file(test_install.cpp test_install.cu COPYONLY)
    add_executable(test_install_cuda ${CMAKE_CURRENT_BINARY_DIR}/test_install.cu)
    target_compile_options(test_install_cuda
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
    # we handle CUDA architecture flags for now, disable CMake handling
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_target_properties(test_install_cuda PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
    target_compile_definitions(test_install_cuda PRIVATE HAS_CUDA=1)
    target_compile_definitions(test_install_cuda PRIVATE HAS_REFERENCE=${HAS_REFERENCE})
    target_link_libraries(test_install_cuda PRIVATE Ginkgo::ginkgo)
endif()

if(GINKGO_BUILD_HIP)
    find_package(HIP REQUIRED)

    set_source_files_properties(test_install.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    # The library was compiled with `-fPIC` when in shared library mode
    if (GINKGO_BUILD_SHARED_LIBS)
        set (GINKGO_PIC_OPTION "-fPIC")
        set (GINKGO_CUDA_PIC_OPTION "-Xcompiler '-fPIC'")
    else()
        set (GINKGO_PIC_OPTION "$<$<CONFIG:Debug>:-fPIC>")
    endif()
    if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        set(TESTINSTALL_CUDA_ARCH_FLAGS "${GINKGO_CUDA_ARCH_FLAGS}")
    endif()
    if (CMAKE_CUDA_HOST_COMPILER)
        set(TESTINSTALL_CUDA_HOST_COMPILER "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
    endif()
    hip_add_executable(test_install_hip test_install.cpp
        HIPCC_OPTIONS "-std=c++14"
        CLANG_OPTIONS "${GINKGO_PIC_OPTION}"
        NVCC_OPTIONS "${GINKGO_CUDA_PIC_OPTION}" "${TESTINSTALL_CUDA_ARCH_FLAGS}" "${TESTINSTALL_CUDA_HOST_COMPILER}")

    target_link_libraries(test_install_hip PRIVATE Ginkgo::ginkgo)
    target_compile_definitions(test_install_hip PRIVATE HAS_HIP=1)
    target_compile_definitions(test_install_hip PRIVATE HAS_REFERENCE=${HAS_REFERENCE})

    # If we always link with CXX there is no RPATH issue
    set_target_properties(test_install_hip PROPERTIES LINKER_LANGUAGE CXX)

    # Instead of using CXX, it's possible to instead use
    # target_link_libraries(test_install_hip PRIVATE ${GINKGO_INSTALL_RPATH_FOR_HIP})
endif()

# Installation step
set(TESTINSTALL_INSTALL_DIR "${GINKGO_INSTALL_PREFIX}/smoke_tests")
set(TESTINSTALL_INSTALL_DATADIR "${TESTINSTALL_INSTALL_DIR}/data")

# Install the binaries and data
install(TARGETS test_install RUNTIME DESTINATION ${TESTINSTALL_INSTALL_DIR})
if(GINKGO_BUILD_CUDA)
    install(TARGETS test_install_cuda RUNTIME DESTINATION ${TESTINSTALL_INSTALL_DIR})
endif()
if(GINKGO_BUILD_HIP)
    install(TARGETS test_install_hip RUNTIME DESTINATION ${TESTINSTALL_INSTALL_DIR})
endif()
install(DIRECTORY "${TestInstall_BINARY_DIR}/data/"
    DESTINATION "${TESTINSTALL_INSTALL_DATADIR}")
