if(GINKGO_BUILD_CUDA)
    enable_language(CUDA)
    if (GINKGO_USE_EXTERNAL_CAS)
        include(CudaArchitectureSelector RESULT_VARIABLE GINKGO_CAS_FILE)
    endif()
    if (NOT GINKGO_CAS_FILE)
        add_subdirectory(CudaArchitectureSelector)
    endif()
endif()

# For each external package, check if they are already available, otherwise use ours
if(GINKGO_BUILD_TESTS)
    ginkgo_add_subdirectory(GTest gtest)
endif()

if(GINKGO_DEVEL_TOOLS)
    set(GCF_IGNORE_LIST "third_party" CACHE STRING "Ignore directories for GCF")
    add_subdirectory(git-cmake-format)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    ginkgo_add_subdirectory(gflags gflags)
    ginkgo_add_subdirectory(RapidJSON rapidjson)
    if(WIN32)
        target_link_libraries(gflags INTERFACE shlwapi.lib)
    endif()
    # RapidJSON does not provide a modern CMake target so far. Let's fix that!
    if(RapidJSON_FOUND)
        add_library(rapidjson INTERFACE)
        target_include_directories(rapidjson INTERFACE ${RapidJSON_INCLUDE_DIRS})
    endif()
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
