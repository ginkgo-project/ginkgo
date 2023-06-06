cmake_minimum_required(VERSION 3.21)
project(kokkos-assembly CXX)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.7.0 REQUIRED)
endif ()

find_package(Kokkos REQUIRED)

message(STATUS "Kokkos_ENABLE_HIP: ${Kokkos_ENABLE_HIP}")

ginkgo_enable_kokkos_language()

ginkgo_add_kokkos_executable(kokkos-assembly1 kokkos_assembly1.cpp)
ginkgo_add_kokkos_executable(kokkos-assembly2 kokkos_assembly2.cpp)
