cmake_minimum_required(VERSION 3.9)
project(batch-block-diagonal-system)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(batch-block-diagonal-system batch-block-diagonal-system.cpp)
target_link_libraries(batch-block-diagonal-system Ginkgo::ginkgo)

