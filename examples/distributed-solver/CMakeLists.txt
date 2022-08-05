cmake_minimum_required(VERSION 3.9)
project(distributed-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

find_package(PkgConfig REQUIRED)

message("EXAMPLE: ${CMAKE_PREFIX_PATH}")
pkg_check_modules(Kahypar REQUIRED IMPORTED_TARGET libkahypar)

add_executable(distributed-solver distributed-solver.cpp)
target_link_libraries(distributed-solver Ginkgo::ginkgo PkgConfig::Kahypar)

file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/km1_kKaHyPar_sea20.ini ${CMAKE_CURRENT_BINARY_DIR}/km1_kKaHyPar_sea20.ini SYMBOLIC)
