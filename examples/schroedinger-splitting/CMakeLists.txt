cmake_minimum_required(VERSION 3.16)
project(schroedinger-splitting)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()
find_package(OpenCV REQUIRED)

add_executable(schroedinger-splitting schroedinger-splitting.cpp)
target_link_libraries(schroedinger-splitting Ginkgo::ginkgo ${OpenCV_LIBS})

# Copy the data files to the execution directory
configure_file(../../matrices/examples/gko_logo_2d.mtx data/gko_logo_2d.mtx COPYONLY)
configure_file(../../matrices/examples/gko_text_2d.mtx data/gko_text_2d.mtx COPYONLY)
