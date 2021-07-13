cmake_minimum_required(VERSION 3.9)
project(standalone-simple-solver)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.4.0 REQUIRED)
endif()

add_executable(standalone-simple-solver standalone-simple-solver.cpp)
target_link_libraries(standalone-simple-solver Ginkgo::ginkgo rapidjson)
target_include_directories(standalone-simple-solver PRIVATE ${CMAKE_SOURCE_DIR}/extension/resource_manager/include)

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
configure_file(data/solver.json data/solver.json COPYONLY)
configure_file(data/A.json data/A.json COPYONLY)
