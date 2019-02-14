include(package_loader.cmake)

add_subdirectory(CudaArchitectureSelector)

if(GINKGO_BUILD_TESTS)
    add_subdirectory(gtest)
endif()

if(GINKGO_DEVEL_TOOLS)
    set(GCF_IGNORE_LIST "third_party" CACHE STRING "Ignore directories for GCF")
    add_subdirectory(git-cmake-format)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    add_subdirectory(gflags)
    add_subdirectory(rapidjson)
endif()

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" PARENT_SCOPE)
