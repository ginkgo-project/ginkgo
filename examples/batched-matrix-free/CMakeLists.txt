cmake_minimum_required(VERSION 3.21)
project(batched-solver LANGUAGES CXX HIP)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(batched-matrix-free batched-matrix-free.cpp apply.cpp
    apply.hip.cpp)
target_link_libraries(batched-matrix-free Ginkgo::ginkgo)
set_source_files_properties(apply.hip.cpp PROPERTIES LANGUAGE HIP)
