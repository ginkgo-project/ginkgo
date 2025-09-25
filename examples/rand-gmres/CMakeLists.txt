cmake_minimum_required(VERSION 3.16)
project(rand-gmres)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.9.0 REQUIRED)
endif()

add_executable(rand-gmres rand-gmres.cpp)
target_link_libraries(rand-gmres Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
