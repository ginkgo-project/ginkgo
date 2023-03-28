cmake_minimum_required(VERSION 3.9)
project(distributed-overlapping-solvers)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    find_package(googletest)
endif ()

add_executable(distributed-overlapping-solver distributed-overlapping-solver.cpp)
target_link_libraries(distributed-overlapping-solver Ginkgo::ginkgo)

add_executable(distributed-overlapping-solver-2d distributed-overlapping-solver-2d.cpp)
target_link_libraries(distributed-overlapping-solver-2d Ginkgo::ginkgo)

add_executable(non-overlapping-solver-2d non-overlapping-solver-2d.cpp)
target_link_libraries(non-overlapping-solver-2d Ginkgo::ginkgo)

add_executable(testing testing.cpp)
target_link_libraries(testing Ginkgo::ginkgo GTest::Main GTest::GTest)
