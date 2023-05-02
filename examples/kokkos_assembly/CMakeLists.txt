cmake_minimum_required(VERSION 3.13)
project(kokkos-assembly CXX HIP)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.7.0 REQUIRED)
endif ()

find_package(Kokkos REQUIRED)

add_executable(kokkos-assembly1 kokkos_assembly1.cpp)
target_link_libraries(kokkos-assembly1 Ginkgo::ginkgo Kokkos::kokkos)
set_target_properties(kokkos-assembly1 PROPERTIES CXX_COMPILER_LAUNCHER "" LANGUAGE HIP)

add_executable(kokkos-assembly2 kokkos_assembly2.cpp)
target_link_libraries(kokkos-assembly2 Ginkgo::ginkgo Kokkos::kokkos)
set_target_properties(kokkos-assembly2 PROPERTIES CXX_COMPILER_LAUNCHER "" LANGUAGE HIP)
