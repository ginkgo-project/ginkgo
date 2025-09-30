# For each external package, check if they are already available, otherwise use ours
if(GINKGO_BUILD_TESTS AND (NOT GTest_FOUND))
    add_subdirectory(gtest)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    if(NOT gflags_FOUND)
        add_subdirectory(gflags)
    endif()
endif()

if(GINKGO_BUILD_BENCHMARKS)
    if(NOT nlohmann_json_schema_validator_FOUND)
        add_subdirectory(json_scheme_validator)
    endif()
endif()

if(GINKGO_BUILD_TESTS OR GINKGO_BUILD_BENCHMARKS OR GINKGO_BUILD_EXAMPLES)
    if(NOT nlohmann_json_FOUND)
        add_subdirectory(nlohmann_json)
    endif()
endif()

if(GINKGO_TEST_NONDEFAULT_STREAM)
    add_subdirectory(identify_stream_usage)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
