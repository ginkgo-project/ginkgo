cmake_minimum_required(VERSION 3.9)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

include(CheckLanguage)
check_language(CUDA)

if(GINKGO_HAVE_PAPI_SDE)
    find_package(PAPI REQUIRED OPTIONAL_COMPONENTS sde)
endif()

# Needed because of a known issue with CUDA while linking statically.
# For details, see https://gitlab.kitware.com/cmake/cmake/issues/18614
if((NOT GINKGO_BUILD_SHARED_LIBS) AND GINKGO_BUILD_CUDA)
    enable_language(CUDA)
endif()

if((NOT GINKGO_BUILD_SHARED_LIBS) AND GINKGO_BUILD_HIP)
    find_package(HIP REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipsparse REQUIRED)
    if(GINKGO_HIP_PLATFORM MATCHES "hcc")
        ginkgo_hip_ban_link_hcflag(hcc::hccrt)
        ginkgo_hip_ban_link_hcflag(hcc::hc_am)
        ginkgo_hip_ban_link_hcflag(hcc::mcwamp)
        ginkgo_hip_ban_compile_hcflag(hcc::hccrt)
        ginkgo_hip_ban_compile_hcflag(hcc::hc_am)
        ginkgo_hip_ban_compile_hcflag(hcc::mcwamp)
    endif()
endif()

add_executable(test_install test_install.cpp)
target_compile_features(test_install PUBLIC cxx_std_11)
target_link_libraries(test_install PRIVATE Ginkgo::ginkgo)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    add_executable(test_install_cuda test_install_cuda.cu)
    target_link_libraries(test_install_cuda PRIVATE Ginkgo::ginkgo)
endif()