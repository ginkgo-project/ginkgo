cmake_minimum_required(VERSION 3.16)
project(kokkos-assembly CXX)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()
if(NOT GINKGO_EXTENSION_KOKKOS)
    message(FATAL_ERROR "The Kokkos examples requires that GINKGO_EXTENSION_KOKKOS=ON is set.")
endif()

find_package(Kokkos 3.7 REQUIRED)

add_executable(kokkos-assembly kokkos_assembly.cpp)
target_link_libraries(kokkos-assembly Ginkgo::ginkgo Kokkos::kokkos)
