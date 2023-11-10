cmake_minimum_required(VERSION 3.16)
project(ginkgo-overhead)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(ginkgo-overhead ginkgo-overhead.cpp)
target_link_libraries(ginkgo-overhead Ginkgo::ginkgo)
