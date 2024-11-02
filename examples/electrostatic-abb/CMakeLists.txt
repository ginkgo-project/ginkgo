cmake_minimum_required(VERSION 3.16)
project(electrostatic-abb)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(electrostatic-abb electrostatic-abb.cpp)
target_link_libraries(electrostatic-abb Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/sphere.amtx data/sphere.amtx COPYONLY)
configure_file(data/sphere.bmtx data/sphere.bmtx COPYONLY)
