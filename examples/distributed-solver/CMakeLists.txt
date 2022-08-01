cmake_minimum_required(VERSION 3.9)
project(distributed-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

find_package(PkgConfig REQUIRED)

pkg_check_modules(Kahypar REQUIRED IMPORTED_TARGET libkahypar)


add_executable(distributed-solver distributed-solver.cpp)
target_link_libraries(distributed-solver Ginkgo::ginkgo PkgConfig::Kahypar parmetis)
