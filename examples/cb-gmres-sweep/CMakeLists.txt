cmake_minimum_required(VERSION 3.16)
project(cb-gmres-sweep)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(cb-gmres-sweep cb-gmres-sweep.cpp)
target_link_libraries(cb-gmres-sweep Ginkgo::ginkgo)
