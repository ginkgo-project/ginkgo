#.rst:
# Ginkgo CTestScript
# -------
#
# Runs our tests through CTest, with support for Coverage or memory checking.
#
# This script provides a full CTest run whith result submission to Ginkgo's
# CDash dashboard. The supported runs are:
# + With or without coverage, requires the gcov tool.
# + With or without address sanitizers.
# + With or without memory sanitizers.
# + With or without thread sanitizers.
# + With or without leak sanitizers.
# + With or without undefined behavior (UB) sanitizers.
# + With or without valgrind, requires the valgrind tool.
#
# Note that only one of these can be ran at once, as the build types
# conflict. Ginkgo is always configured with CUDA, HIP, OpenMP and Reference
# support, except for ThreadSanitizer, AddressSanitizer, LeakSanitizer,
# UndefinedBehaviorSanitizer builds. The results are always sent to the
# dashboard: https://my.cdash.org/index.php?project=Ginkgo+Project
#
# Running the script
# ^^^^^^^^^^^^^^^^^^
#
# To run the script, launch the command `ctest -S cmake/CTestScript.cmake`
# from the Ginkgo's source directory. The default settings are for the CI
# system's DEBUG tests run. To configure the script use standard CMake `-D`
# parameters. For example, the following command runs coverage.
#
# `ctest -S cmake/CTestScript.cmake -DCTEST_BUILD_CONFIGURATION=COVERAGE`
#
# Instead, this runs the ThreadSanitizer:
#
# `ctest -S cmake/CTestScript.cmake -DCTEST_BUILD_CONFIGURATION=TSAN
#     -DCTEST_MEMORYCHECK_TYPE=ThreadSanitizer`
#
# Input Variables
# ^^^^^^^^^^^^^^^^
#
# This script can be configured with the following input variables:
#
# ``CTEST_SOURCE_DIRECTORY``
# Where the sources are located. By default, the current directory.
#
# ``CTEST_BINARY_DIRECTORY``
# In which directory should the sources be builts. Default, `./build`.
#
# ``CTEST_SITE``
# A string to describe the machine this is ran on. Default FineCI.
#
# ``CTEST_CMAKE_GENERATOR``
# Which generator should be used for the build. Default `Ninja`, except
# for COVERAGE builds where `Unix Makefiles` is used.
#
# ``CTEST_BUILD_CONFIGURATION``
# Which configuration should Ginkgo be built with. Default `DEBUG`.
# The supported values are: COVERAGE, TSAN, UBSAN, DEBUG, and
# RELEASE.
#
# ``CTEST_TEST_MODEL``
# Which CTest test model should be used. Default `Continuous`.
# The supported values are the same as CTest's, namely:
# Experimental, Nightly, Continuous.
#
# ``CTEST_BUILD_NAME``
# The name of the build being ran. Default: `CTEST_BUILD_CONFIGURATION`
#
# ``CTEST_MEMORYCHECK_TYPE``
# Whether memorycheck should be ran. Default: `NONE`. Supported values are:
# Valgrind, AddressSanitizer, LeakSanitizer, ThreadSanitizer,
# UndefinedBehaviorSanitizer and NONE.
#

if (NOT DEFINED CTEST_SOURCE_DIRECTORY)
    set(CTEST_SOURCE_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/..")
endif()

if (NOT DEFINED CTEST_BINARY_DIRECTORY)
    set(CTEST_BINARY_DIRECTORY "${CTEST_SOURCE_DIRECTORY}/build")
endif()

if (NOT DEFINED CTEST_SITE)
    set(CTEST_SITE "Linux-FineCI")
endif()

if (NOT DEFINED CTEST_CMAKE_GENERATOR)
    if (CTEST_BUILD_CONFIGURATION STREQUAL "COVERAGE")
        set(CTEST_CMAKE_GENERATOR "Unix Makefiles")
    else()
        set(CTEST_CMAKE_GENERATOR "Ninja")
    endif()
endif()

# Supported: COVERAGE, ASAN, LSAN, TSAN, UBSAN, DEBUG and RELEASE
if (NOT DEFINED CTEST_BUILD_CONFIGURATION)
    set(CTEST_BUILD_CONFIGURATION "DEBUG")
endif()

if (NOT DEFINED CTEST_TEST_MODEL)
    set(CTEST_TEST_MODEL "Continuous")
endif()

if (NOT DEFINED CTEST_BUILD_NAME)
    set(CTEST_BUILD_NAME "${CTEST_BUILD_CONFIGURATION}")
endif()

#Supported: Valgrind, ThreadSanitizer, AddressSanitizer, LeakSanitizer
#and UndefinedBehaviorSanitizer.
if (NOT DEFINED CTEST_MEMORYCHECK_TYPE)
    set(CTEST_MEMORYCHECK_TYPE "NONE")
endif()

# Find coverage and valgrind tools
if(CTEST_MEMORYCHECK_TYPE STREQUAL "Valgrind")
    find_program(CTEST_MEMORYCHECK_COMMAND valgrind)
    set(CTEST_BUILD_NAME "Valgrind")
    set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--trace-children=yes --leak-check=full")
    set(CTEST_MEMORYCHECK_SUPPRESSIONS_FILE "${CTEST_SOURCE_DIRECTORY}/dev_tools/valgrind/suppressions.supp")
endif()

if(CTEST_MEMORYCHECK_TYPE STREQUAL "CudaMemcheck")
    find_program(CTEST_MEMORYCHECK_COMMAND cuda-memcheck)
    set(CTEST_BUILD_NAME "CudaMemcheck")
endif()

if(CTEST_BUILD_CONFIGURATION STREQUAL "COVERAGE")
    find_program(CTEST_COVERAGE_COMMAND gcov)
endif()

if(NOT CTEST_MEMORYCHECK_TYPE STREQUAL "Valgrind" AND NOT CTEST_MEMORYCHECK_TYPE STREQUAL "CudaMemcheck")
    set(CTEST_MEMORYCHECK_SANITIZER_OPTIONS "${CTEST_MEMORYCHECK_SANITIZER_OPTIONS}:allocator_may_return_null=1:verbosity=1")
endif()

include(ProcessorCount)
ProcessorCount(PROC_COUNT)
if(NOT PROC_COUNT EQUAL 0)
    if (DEFINED ENV{CI_PARALLELISM})
        set(PROC_COUNT "$ENV{CI_PARALLELISM}")
    elseif(PROC_COUNT LESS 4)
        set(PROC_COUNT 1)
    else()
        set(PROC_COUNT 4)
    endif()
    if(NOT WIN32)
        set(CTEST_BUILD_FLAGS "-j${PROC_COUNT}")
    endif(NOT WIN32)
endif()


ctest_start("${CTEST_TEST_MODEL}")
ctest_submit(PARTS Start)

if (CTEST_MEMORYCHECK_TYPE STREQUAL "CudaMemcheck")
    # generate line number information for CUDA
    set(GINKGO_CONFIGURE_OPTIONS "-DGINKGO_DEVEL_TOOLS=OFF;-DGINKGO_BUILD_REFERENCE=ON;-DGINKGO_BUILD_OMP=OFF;-DGINKGO_BUILD_CUDA=ON;-DGINKGO_BUILD_HIP=ON;-DCMAKE_BUILD_TYPE=${CTEST_BUILD_CONFIGURATION};-DCMAKE_CUDA_FLAGS=-lineinfo")
elseif((NOT CTEST_MEMORYCHECK_TYPE STREQUAL "NONE" AND NOT CTEST_MEMORYCHECK_TYPE STREQUAL "Valgrind") OR CTEST_BUILD_CONFIGURATION STREQUAL "COVERAGE")
    set(GINKGO_CONFIGURE_OPTIONS "-DGINKGO_DEVEL_TOOLS=OFF;-DGINKGO_BUILD_REFERENCE=ON;-DGINKGO_BUILD_OMP=ON;-DGINKGO_BUILD_CUDA=OFF;-DGINKGO_BUILD_HIP=OFF;-DCMAKE_BUILD_TYPE=${CTEST_BUILD_CONFIGURATION}")
else()
    set(GINKGO_CONFIGURE_OPTIONS "-DGINKGO_DEVEL_TOOLS=OFF;-DGINKGO_BUILD_REFERENCE=ON;-DGINKGO_BUILD_OMP=ON;-DGINKGO_BUILD_CUDA=ON;-DGINKGO_BUILD_HIP=ON;-DCMAKE_BUILD_TYPE=${CTEST_BUILD_CONFIGURATION}")
endif()

# UBSAN needs gold linker
if (CTEST_MEMORYCHECK_TYPE STREQUAL "UndefinedBehaviorSanitizer")
    set(GINKGO_CONFIGURE_OPTIONS "${GINKGO_CONFIGURE_OPTIONS};-DCMAKE_SHARED_LINKER_FLAGS=-fuse-ld=gold;-DCMAKE_EXE_LINKER_FLAGS=-fuse-ld=gold")
endif()

ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}" OPTIONS "${GINKGO_CONFIGURE_OPTIONS}" APPEND)
ctest_submit(PARTS Configure)

ctest_read_custom_files( ${CTEST_BINARY_DIRECTORY} )

if (DEFINED GINKGO_SONARQUBE_TEST)
    set(CTEST_BUILD_COMMAND "build-wrapper-linux-x86-64 --out-dir bw-output make -j${PROC_COUNT}")
endif()
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}" APPEND)
ctest_submit(PARTS Build)


if (CTEST_MEMORYCHECK_TYPE STREQUAL "NONE")
    ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}" APPEND)
    ctest_submit(PARTS Test)
endif()

if (CTEST_BUILD_CONFIGURATION STREQUAL "COVERAGE")
    ctest_coverage(BUILD "${CTEST_BINARY_DIRECTORY}" APPEND)
    ctest_submit(PARTS Coverage)
endif()

if(NOT CTEST_MEMORYCHECK_TYPE STREQUAL "NONE")
    if(CTEST_MEMORYCHECK_TYPE STREQUAL "CudaMemcheck")
        ctest_memcheck(BUILD "${CTEST_BINARY_DIRECTORY}" INCLUDE "^(cuda|hip).*" APPEND)
    else()
        ctest_memcheck(BUILD "${CTEST_BINARY_DIRECTORY}" APPEND)
    endif()
    ctest_submit(PARTS MemCheck)
endif()
