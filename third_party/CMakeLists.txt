# For each external package, check if they are already available, otherwise use ours
if(GINKGO_BUILD_TESTS AND (NOT GTest_FOUND))
    add_subdirectory(gtest)
endif()
if(GINKGO_BUILD_MPI)
    add_library(gtest_mpi_main "")
    target_sources(gtest_mpi_main
        PRIVATE
        gtest/gtest_mpi_listener.cpp)
    find_package(MPI REQUIRED)
    target_link_libraries(gtest_mpi_main PRIVATE GTest::GTest MPI::MPI_CXX)
    add_library(GTest::MPI_main ALIAS gtest_mpi_main)
endif()

if(GINKGO_BUILD_HWLOC AND (NOT HWLOC_FOUND))
    add_subdirectory(hwloc)
endif()

if(GINKGO_DEVEL_TOOLS)
    set(GCF_IGNORE_LIST "third_party" CACHE STRING "Ignore directories for GCF")
    add_subdirectory(git-cmake-format)
else()
    add_subdirectory(dummy-hook)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    if (NOT gflags_FOUND)
        add_subdirectory(gflags)
    endif()
    if (NOT RapidJSON_FOUND)
        add_subdirectory(rapidjson)
    endif()
endif()

if (GINKGO_TEST_NONDEFAULT_STREAM)
    add_subdirectory(identify_stream_usage)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
