cmake_minimum_required(VERSION 3.16)
project(papi-logging)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()

if (NOT GINKGO_HAVE_PAPI_SDE)
    message(FATAL_ERROR "This example needs Ginkgo built with PAPI support")
endif()

add_executable(papi-logging papi-logging.cpp)
target_link_libraries(papi-logging ginkgo PAPI::PAPI)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
