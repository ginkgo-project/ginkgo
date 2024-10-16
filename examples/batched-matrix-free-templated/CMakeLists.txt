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

add_executable(batched-matrix-free-templated batched-matrix-free-templated.cpp)
target_link_libraries(batched-matrix-free-templated Ginkgo::ginkgo cxxopts::cxxopts)
target_compile_features(batched-matrix-free-templated PRIVATE cxx_std_17 hip_std_17 cuda_std_17)
if (GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    set_source_files_properties(batched-matrix-free-templated.cpp PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(batched-matrix-free-templated PRIVATE GKO_COMPILING_CUDA=1 GKO_DEVICE_NAMESPACE=cuda)
    target_compile_options(batched-matrix-free-templated PRIVATE --expt-relaxed-constexpr)
elseif (GINKGO_BUILD_HIP)
    enable_language(HIP)
    find_package(hip REQUIRED)
    target_link_libraries(batched-matrix-free-templated hip::host)
    set_source_files_properties(batched-matrix-free-templated.cpp PROPERTIES LANGUAGE HIP)
    target_compile_definitions(batched-matrix-free-templated PRIVATE GKO_COMPILING_HIP=1 GKO_DEVICE_NAMESPACE=hip)
endif ()
