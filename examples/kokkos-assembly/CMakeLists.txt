cmake_minimum_required(VERSION 3.16)
project(kokkos-assembly CXX)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 2.0.0 REQUIRED)
endif()
find_package(Kokkos 4.1.00 REQUIRED)

# Kokkos doesn't handle any compiler launcher well, so it's disable it
unset(CMAKE_CXX_COMPILER_LAUNCHER)

add_executable(kokkos-assembly kokkos-assembly.cpp)
target_link_libraries(kokkos-assembly Ginkgo::ginkgo Kokkos::kokkos)
