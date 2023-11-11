cmake_minimum_required(VERSION 3.16)
project(kokkos-assembly CXX)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

find_package(Kokkos REQUIRED)

add_executable(kokkos-assembly kokkos_assembly.cpp)
target_link_libraries(kokkos-assembly Ginkgo::ginkgo Kokkos::kokkos)
