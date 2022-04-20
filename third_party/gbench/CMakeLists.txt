# ginkgo_load_git_package(gbench_external
#         "https://github.com/google/benchmark.git"
#         "73d4d5e8d6d449fc8663765a42aa8aeeee844489"
#         "-DCMAKE_CXX_FLAGS=-fPIC"
#         "-DBENCHMARK_ENABLE_TESTING=OFF"
#         "-DCMAKE_CXX_FLAGS_DEBUG=${CMAKE_CXX_FLAGS_DEBUG}"
#         "-DCMAKE_CXX_FLAGS_RELEASE=${CMAKE_CXX_FLAGS_RELEASE}"
#         "-DCMAKE_C_FLAGS_DEBUG=${CMAKE_C_FLAGS_DEBUG}"
#         "-DCMAKE_C_FLAGS_RELEASE=${CMAKE_C_FLAGS_RELEASE}")

# ginkgo_add_external_target(GBench::GBench benchmark src/include build/src
#    STATIC "d" gbench_external FALSE)
# ginkgo_add_external_target(GBench::Main benchmark_main src/include build/src
#     STATIC "d" gbench_external FALSE)

message(STATUS "Fetching external GBench")
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        73d4d5e8d6d449fc8663765a42aa8aeeee844489
)
FetchContent_GetProperties(googlebenchmark)
if(NOT googlebenchmark_POPULATED)
    FetchContent_Populate(googlebenchmark)
    add_subdirectory(${googlebenchmark_SOURCE_DIR} ${googlebenchmark_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
# make sure the tests DLLs are placed in the working path for CTest
set_target_properties(benchmark benchmark_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
    ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
    LIBRARY_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}")
# by default, the outdated targets are not being exported
add_library(GBench::Main ALIAS benchmark_main)
add_library(GBench::GBench ALIAS benchmark)
