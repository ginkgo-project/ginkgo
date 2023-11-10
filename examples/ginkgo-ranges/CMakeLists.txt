cmake_minimum_required(VERSION 3.16)
project(ginkgo-ranges)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()
add_executable(ginkgo-ranges ginkgo-ranges.cpp)
target_link_libraries(ginkgo-ranges Ginkgo::ginkgo)
