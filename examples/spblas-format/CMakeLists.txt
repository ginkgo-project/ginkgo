cmake_minimum_required(VERSION 3.20)
project(spblas-format CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.8.0 REQUIRED)
endif()





include(FetchContent)
FetchContent_Declare(
  spblas
  GIT_REPOSITORY https://github.com/SparseBLAS/spblas-reference.git
  GIT_TAG main)
FetchContent_MakeAvailable(spblas)


add_executable(spblas-format spblas-format.cpp)
target_link_libraries(spblas-format Ginkgo::ginkgo spblas)

