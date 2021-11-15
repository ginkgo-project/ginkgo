# For each external package, check if they are already available, otherwise use ours
if(GINKGO_BUILD_TESTS AND (NOT GTest_FOUND))
    add_subdirectory(gtest)
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
endif()
if(GINKGO_BUILD_BENCHMARKS OR GINKGO_BUILD_EXTENSION)
    if (NOT RapidJSON_FOUND)
        add_subdirectory(rapidjson)
    endif()
endif()

if (GINKGO_TEST_NONDEFAULT_STREAM)
    add_subdirectory(identify_stream_usage)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
