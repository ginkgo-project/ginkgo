cmake_minimum_required(VERSION 3.21)
project(batched-solver LANGUAGES CXX)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif ()

include(FetchContent)
FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG        v3.2.0
)
FetchContent_MakeAvailable(cxxopts)

find_package(Kokkos 4.1 REQUIRED)

# This is taken from the kokkos extension test CMake
function(_ginkgo_check_build_config _build)
    string(TOUPPER ${_build} _build)
    if(NOT GINKGO_BUILD_${_build})
        message(FATAL_ERROR "Building example with ${_build} enabled for Kokkos, but not for Ginkgo.")
    endif()
    unset(_build)
endfunction()

set(definitions)
if(Kokkos_ENABLE_CUDA)
    _ginkgo_check_build_config(cuda)
    set(definitions GKO_COMPILING_CUDA=1 GKO_DEVICE_NAMESPACE=cuda)
    enable_language(CUDA)
elseif(Kokkos_ENABLE_HIP)
    _ginkgo_check_build_config(hip)
    set(definitions GKO_COMPILING_HIP=1 GKO_DEVICE_NAMESPACE=hip)
    enable_language(HIP)
    find_package(hip REQUIRED)
elseif(Kokkos_ENABLE_SYCL)
    _ginkgo_check_build_config(sycl)
    set(definitions GKO_COMPILING_DPCPP=1 GKO_DEVICE_NAMESPACE=dpcpp)
endif()
if(Kokkos_ENABLE_OPENMP)
    _ginkgo_check_build_config(omp)
endif ()
if (Kokkos_ENABLE_SERIAL)
    _ginkgo_check_build_config(reference)
endif()

add_executable(batched-matrix-free-templated batched-matrix-free-templated.cpp)
target_compile_features(batched-matrix-free-templated PRIVATE cxx_std_17 hip_std_17 cuda_std_17)
target_compile_definitions(batched-matrix-free-templated PRIVATE ${definitions})
target_link_libraries(batched-matrix-free-templated Ginkgo::ginkgo cxxopts::cxxopts Kokkos::kokkos)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif ()
