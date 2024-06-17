cmake_minimum_required(VERSION 3.16)
project(getting-started)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(getting-started getting-started.cpp)
target_link_libraries(getting-started Ginkgo::ginkgo)
