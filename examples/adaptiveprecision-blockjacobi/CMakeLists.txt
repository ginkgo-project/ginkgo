cmake_minimum_required(VERSION 3.16)
project(adaptiveprecision-blockjacobi)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(adaptiveprecision-blockjacobi adaptiveprecision-blockjacobi.cpp)
target_link_libraries(adaptiveprecision-blockjacobi Ginkgo::ginkgo)

configure_file(data/A.mtx data/A.mtx COPYONLY)
