cmake_minimum_required(VERSION 3.16)
project(iterative-refinement)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

add_executable(iterative-refinement iterative-refinement.cpp)
target_link_libraries(iterative-refinement Ginkgo::ginkgo)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
