cmake_minimum_required(VERSION 3.16)
project(mixed-precision-ir)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(mixed-precision-ir mixed-precision-ir.cpp)
target_link_libraries(mixed-precision-ir Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
