message(STATUS "Fetching external GTest")
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.12.1
)
# need to set the variables in CACHE due to CMP0077
set(gtest_disable_pthreads ON CACHE INTERNAL "")
if(MSVC)
    set(gtest_force_shared_crt ON CACHE INTERNAL "")
endif()
FetchContent_GetProperties(googletest)
if(NOT googletest_POPULATED)
    FetchContent_Populate(googletest)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
# make sure the tests DLLs are placed in the working path for CTest
set_target_properties(gtest gtest_main PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
    ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}"
    LIBRARY_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}")
# If CXX compiler is dpcpp, use -ffp-model=precise
# Otherwise, it will throw src/gtest.cc:1583:8: error: comparison with NaN always evaluates to false in fast floating point modes
if(CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
    target_compile_options(gtest PRIVATE "-ffp-model=precise")
    target_compile_options(gtest_main PRIVATE "-ffp-model=precise")
endif()
# by default, the outdated targets are not being exported
add_library(GTest::Main ALIAS gtest_main)
add_library(GTest::GTest ALIAS gtest)
