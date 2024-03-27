# For each external package, check if they are already available, otherwise use ours
if(GINKGO_BUILD_TESTS AND (NOT GTest_FOUND))
    add_subdirectory(gtest)
endif()

if(NOT GINKGO_DEVEL_TOOLS)
    add_subdirectory(dummy-hook)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    if (NOT gflags_FOUND)
        add_subdirectory(gflags)
    endif()
    if (NOT nlohmann_json_FOUND)
        add_subdirectory(nlohmann_json)
    endif()
endif()

if (GINKGO_TEST_NONDEFAULT_STREAM)
    add_subdirectory(identify_stream_usage)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
