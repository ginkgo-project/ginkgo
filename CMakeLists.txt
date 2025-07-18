cmake_minimum_required(VERSION 3.16)

project(
    Ginkgo
    LANGUAGES CXX
    VERSION 2.0.0
    DESCRIPTION
        "A numerical linear algebra library targeting many-core architectures"
)
set(Ginkgo_VERSION_TAG "develop")
set(PROJECT_VERSION_TAG ${Ginkgo_VERSION_TAG})
if(Ginkgo_VERSION_TAG STREQUAL "master")
    set(GINKGO_VERSION_TAG_DEPRECATED ON)
else()
    set(GINKGO_VERSION_TAG_DEPRECATED OFF)
endif()
if(GINKGO_VERSION_TAG_DEPRECATED)
    message(
        WARNING
        "The branch ${Ginkgo_VERSION_TAG} is deprecated and will stop receiving updates after 2025. "
        "Please use the main branch for the latest release, or the develop branch for the latest development updates."
    )
endif()
# Cuda and Hip also look for Threads. Set it before any find_package to ensure the Threads setting is not changed.
set(THREADS_PREFER_PTHREAD_FLAG ON)

# Determine which modules can be compiled
include(cmake/autodetect_executors.cmake)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules/")
include(cmake/autodetect_system_libs.cmake)

# rename helper
include(cmake/rename.cmake)

# Ginkgo configuration options
option(GINKGO_DEVEL_TOOLS "Add development tools to the build system" OFF)
option(GINKGO_BUILD_TESTS "Generate build files for unit tests" ON)
option(GINKGO_BUILD_EXAMPLES "Build Ginkgo's examples" ON)
option(GINKGO_BUILD_BENCHMARKS "Build Ginkgo's benchmarks" ON)
option(GINKGO_BUILD_REFERENCE "Compile reference CPU kernels" ON)
option(GINKGO_BUILD_OMP "Compile OpenMP kernels for CPU" ${GINKGO_HAS_OMP})
option(GINKGO_BUILD_MPI "Compile the MPI module" ${GINKGO_HAS_MPI})
gko_rename_cache(
    GINKGO_BUILD_DPCPP
    GINKGO_BUILD_SYCL
    BOOL
    "Compile SYCL kernels for Intel GPUs or other SYCL enabled hardware"
)
option(
    GINKGO_BUILD_SYCL
    "Compile SYCL kernels for Intel GPUs or other SYCL enabled hardware"
    ${GINKGO_HAS_SYCL}
)
option(GINKGO_BUILD_CUDA "Compile kernels for NVIDIA GPUs" ${GINKGO_HAS_CUDA})
option(
    GINKGO_BUILD_HIP
    "Compile kernels for AMD or NVIDIA GPUs"
    ${GINKGO_HAS_HIP}
)
option(GINKGO_BUILD_DOC "Generate documentation" OFF)
option(
    GINKGO_FAST_TESTS
    "Reduces the input size for a few tests known to be time-intensive"
    OFF
)
option(
    GINKGO_TEST_NONDEFAULT_STREAM
    "Uses non-default streams in CUDA and HIP tests"
    OFF
)
option(
    GINKGO_MIXED_PRECISION
    "Instantiate true mixed-precision kernels (otherwise they will be conversion-based using implicit temporary storage)"
    OFF
)
option(GINKGO_ENABLE_HALF "Enable the use of half precision" ON)
option(GINKGO_ENABLE_BFLOAT16 "Enable the use of bfloat16 precision" ON)
# We do not support half precision in MSVC and msys2 (MINGW).
if(MSVC OR MINGW)
    message(
        STATUS
        "We do not support half/bfloat16 precision in MSVC and MINGW."
    )
    set(GINKGO_ENABLE_HALF
        OFF
        CACHE BOOL
        "Enable the use of half precision"
        FORCE
    )
    set(GINKGO_ENABLE_BFLOAT16
        OFF
        CACHE BOOL
        "Enable the use of bfloat16 precision"
        FORCE
    )
endif()

option(
    GINKGO_SKIP_DEPENDENCY_UPDATE
    "Do not update dependencies each time the project is rebuilt"
    ON
)
option(
    GINKGO_WITH_CLANG_TIDY
    "Make Ginkgo call `clang-tidy` to find programming issues."
    OFF
)
option(
    GINKGO_WITH_IWYU
    "Make Ginkgo call `iwyu` (Include What You Use) to find include issues."
    OFF
)
option(
    GINKGO_WITH_CCACHE
    "Use ccache if available to speed up C++ and CUDA rebuilds by caching compilations."
    ON
)
option(
    GINKGO_CHECK_CIRCULAR_DEPS
    "Enable compile-time checks detecting circular dependencies between libraries and non-self-sufficient headers."
    OFF
)
option(
    GINKGO_CONFIG_LOG_DETAILED
    "Enable printing of detailed configuration log to screen in addition to the writing of files,"
    OFF
)
option(
    GINKGO_BENCHMARK_ENABLE_TUNING
    "Enable tuning variables in the benchmarks. For specific use cases, manual code changes could be required."
    OFF
)
set(GINKGO_VERBOSE_LEVEL
    "1"
    CACHE STRING
    "Verbosity level. Put 0 to turn off. 1 activates a few important messages."
)
set(GINKGO_CUDA_ARCHITECTURES
    "Auto"
    CACHE STRING
    "A list of target NVIDIA GPU architectures. See README.md for more detail."
)
# the details of fine/coarse grain memory and unsafe atomic are available https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#floating-point-fp-atomic-operations-and-coarse-fine-grained-memory-allocations
option(
    GINKGO_HIP_AMD_UNSAFE_ATOMIC
    "Compiler uses unsafe floating point atomic (only for AMD GPU and ROCM >= 5). Default is ON because we use hipMalloc, which is always on coarse grain. Must turn off when allocating memory on fine grain"
    ON
)
option(
    GINKGO_SPLIT_TEMPLATE_INSTANTIATIONS
    "Split template instantiations for slow-to-compile files. This improves parallel build performance"
    ON
)
mark_as_advanced(GINKGO_SPLIT_TEMPLATE_INSTANTIATIONS)
option(
    GINKGO_JACOBI_FULL_OPTIMIZATIONS
    "Use all the optimizations for the CUDA Jacobi algorithm"
    OFF
)
option(BUILD_SHARED_LIBS "Build shared (.so, .dylib, .dll) libraries" ON)
option(GINKGO_BUILD_HWLOC "Build Ginkgo with HWLOC. Default is OFF." OFF)
option(
    GINKGO_BUILD_PAPI_SDE
    "Build Ginkgo with PAPI SDE. Enabled if a system installation is found."
    ${PAPI_SDE_FOUND}
)
option(
    GINKGO_DPCPP_SINGLE_MODE
    "Do not compile double kernels for the DPC++ backend."
    OFF
)
option(GINKGO_INSTALL_RPATH "Set the RPATH when installing its libraries." ON)
option(
    GINKGO_INSTALL_RPATH_ORIGIN
    "Add $ORIGIN (Linux) or @loader_path (MacOS) to the installation RPATH."
    ON
)
option(
    GINKGO_INSTALL_RPATH_DEPENDENCIES
    "Add dependencies to the installation RPATH."
    OFF
)
option(
    GINKGO_FORCE_GPU_AWARE_MPI
    "Assert that the MPI library is GPU aware. This forces Ginkgo to assume that GPU aware functionality is available (OFF (default) or ON), but may fail
     catastrophically in case the MPI implementation is not GPU Aware, and GPU aware functionality has been forced"
    OFF
)
set(GINKGO_CI_TEST_OMP_PARALLELISM
    "4"
    CACHE STRING
    "The number of OpenMP threads to use for a test binary during CTest resource file-constrained test."
)
option(
    GINKGO_EXTENSION_KOKKOS_CHECK_TYPE_ALIGNMENT
    "Enables mapping to Kokkos types to check the alignment of the source and target type."
    ON
)
gko_rename_cache(
    GINKGO_COMPILER_FLAGS
    CMAKE_CXX_FLAGS
    BOOL
    "Flags used by the CXX compiler during all build types."
)
gko_rename_cache(
    GINKGO_CUDA_COMPILER_FLAGS
    CMAKE_CUDA_FLAGS
    BOOL
    "Flags used by the CUDA compiler during all build types."
)

# load executor-specific configuration
if(GINKGO_BUILD_CUDA)
    include(cmake/cuda.cmake)
    if(CUDAToolkit_VERSION VERSION_LESS 11.6)
        message(
            STATUS
            "Disable custom thrust namespace for cuda before 11.6 because it has no effect in the thrust shipped by cuda before 11.6"
        )
        set(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE OFF)
    else()
        message(STATUS "Enable custom thrust namespace for cuda")
        set(GINKGO_CUDA_CUSTOM_THRUST_NAMESPACE ON)
    endif()
endif()
if(GINKGO_BUILD_HIP)
    include(cmake/hip.cmake)
    if(GINKGO_HIP_PLATFORM_AMD AND GINKGO_HIP_VERSION VERSION_LESS 5.7)
        # Hip allow custom namespace but does not fully make everything in the custom namespace before rocm-5.7
        # more specific pr: https://github.com/ROCm/rocThrust/pull/286
        message(
            STATUS
            "Disable custom thrust namespace for hip before 5.7 because hip does not fully support it before 5.7"
        )
        set(GINKGO_HIP_CUSTOM_THRUST_NAMESPACE OFF)
    else()
        message(STATUS "Enable custom thrust namespace for hip")
        set(GINKGO_HIP_CUSTOM_THRUST_NAMESPACE ON)
    endif()
endif()
if(GINKGO_BUILD_SYCL)
    include(cmake/sycl.cmake)
endif()
if(GINKGO_BUILD_OMP)
    find_package(OpenMP 3.0 REQUIRED)
endif()

find_package(Threads REQUIRED)
include(cmake/build_type_helpers.cmake)

# Load other CMake helpers
include(cmake/build_helpers.cmake)
include(cmake/install_helpers.cmake)
include(cmake/compiler_features.cmake)
include(cmake/generate_ginkgo_hpp.cmake)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif()
if(MINGW OR CYGWIN)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        # Otherwise, dynamic_cast to the class marked by final will be failed.
        # https://reviews.llvm.org/D154658 should be relevant
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-assume-unique-vtables")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wa,-mbig-obj")
    endif()
endif()

# For now, PGI/NVHPC nvc++ compiler doesn't seem to support
# `#pragma omp declare reduction`
#
# The math with optimization level -O2 doesn't follow IEEE standard, so we
# enable that back as well.
if(CMAKE_CXX_COMPILER_ID MATCHES "PGI|NVHPC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Kieee")
endif()

set(GINKGO_CIRCULAR_DEPS_FLAGS "-Wl,--no-undefined")

# Use ccache as compilation launcher
if(GINKGO_WITH_CCACHE)
    find_program(CCACHE_PROGRAM ccache)
    if(CCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        if(GINKGO_BUILD_CUDA)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
        endif()
    endif()
endif()

if(GINKGO_BENCHMARK_ENABLE_TUNING)
    # In this state, the tests and examples cannot be compiled without extra
    # complexity/intrusiveness, so we simply disable them.
    set(GINKGO_BUILD_TESTS OFF)
    set(GINKGO_BUILD_EXAMPLES OFF)
endif()

if(GINKGO_BUILD_TESTS)
    message(STATUS "GINKGO_BUILD_TESTS is ON, enabling GINKGO_BUILD_REFERENCE")
    set(GINKGO_BUILD_REFERENCE
        ON
        CACHE BOOL
        "Compile reference CPU kernels"
        FORCE
    )
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif()

# Ensure we have a debug postfix
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
endif()

if(GINKGO_BUILD_TESTS)
    # Configure CTest
    configure_file(
        ${CMAKE_CURRENT_LIST_DIR}/cmake/CTestCustom.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/CTestCustom.cmake
        @ONLY
    )

    # For testing, we need some special matrices
    add_subdirectory(matrices)

    enable_testing()
    include(CTest)
    add_custom_target(quick_test "${CMAKE_CTEST_COMMAND}" -R 'core|reference')
endif()

if(GINKGO_WITH_CLANG_TIDY)
    find_program(GINKGO_CLANG_TIDY_PATH clang-tidy)
endif()

if(GINKGO_WITH_IWYU)
    find_program(GINKGO_IWYU_PATH iwyu)
endif()

# Find important header files, store the definitions in
# include/ginkgo/config.h.in For details, see
# https://gitlab.kitware.com/cmake/community/wikis/doc/tutorials/How-To-Write-Platform-Checks
include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h GKO_HAVE_CXXABI_H)

# Automatically find TAU
set(GINKGO_HAVE_TAU 0)
find_package(PerfStubs QUIET)
if(PerfStubs_FOUND)
    set(GINKGO_HAVE_TAU 1)
endif()
# Automatically find VTune
set(GINKGO_HAVE_VTUNE 0)
find_package(VTune)
if(VTune_FOUND)
    set(GINKGO_HAVE_VTUNE 1)
endif()
# Automatically find METIS
set(GINKGO_HAVE_METIS 0)
find_package(METIS)
if(METIS_FOUND)
    set(GINKGO_HAVE_METIS 1)
endif()
# Automatically detect ROCTX (see hip.cmake)
set(GINKGO_HAVE_ROCTX 0)
if(GINKGO_BUILD_HIP AND ROCTX_FOUND)
    set(GINKGO_HAVE_ROCTX 1)
endif()

# Switch off HWLOC for Windows and MacOS
if(GINKGO_BUILD_HWLOC AND (MSVC OR WIN32 OR CYGWIN OR APPLE))
    set(GINKGO_BUILD_HWLOC
        OFF
        CACHE BOOL
        "Build Ginkgo with HWLOC. Default is OFF. Ginkgo does not support HWLOC on Windows/MacOS"
        FORCE
    )
    message(
        WARNING
        "Ginkgo does not support HWLOC on Windows/MacOS, switch GINKGO_BUILD_HWLOC to OFF"
    )
endif()

set(GINKGO_HAVE_GPU_AWARE_MPI OFF)
set(GINKGO_HAVE_OPENMPI_PRE_4_1_X OFF)
if(GINKGO_BUILD_MPI)
    find_package(MPI 3.1 COMPONENTS CXX REQUIRED)
    if(GINKGO_FORCE_GPU_AWARE_MPI)
        set(GINKGO_HAVE_GPU_AWARE_MPI ON)
    else()
        set(GINKGO_HAVE_GPU_AWARE_MPI OFF)
    endif()

    # use try_compile instead of try_run to prevent cross-compiling issues
    try_compile(
        uses_openmpi
        ${Ginkgo_BINARY_DIR}
        ${Ginkgo_SOURCE_DIR}/cmake/openmpi_test.cpp
        COMPILE_DEFINITIONS -DCHECK_HAS_OPEN_MPI=1
        LINK_LIBRARIES MPI::MPI_CXX
    )
    if(uses_openmpi)
        try_compile(
            valid_openmpi_version
            ${Ginkgo_BINARY_DIR}
            ${Ginkgo_SOURCE_DIR}/cmake/openmpi_test.cpp
            COMPILE_DEFINITIONS -DCHECK_OPEN_MPI_VERSION=1
            LINK_LIBRARIES MPI::MPI_CXX
        )
        if(NOT valid_openmpi_version)
            message(
                WARNING
                "OpenMPI v4.0.x has several bugs that forces us to use non-optimal communication in our distributed "
                "matrix class. To enable faster, non-blocking communication, consider updating your OpenMPI version or "
                "switch to a different vendor."
            )
            set(GINKGO_HAVE_OPENMPI_PRE_4_1_X ON)
        endif()
        unset(valid_openmpi_version)
    endif()
    unset(uses_openmpi)
endif()

# Try to find the third party packages before using our subdirectories
if(GINKGO_BUILD_TESTS)
    find_package(GTest 1.10.0) # No need for QUIET as CMake ships FindGTest
endif()
if(GINKGO_BUILD_BENCHMARKS)
    find_package(gflags 2.2.2 QUIET)
endif()
if(GINKGO_BUILD_TESTS OR GINKGO_BUILD_BENCHMARKS OR GINKGO_BUILD_EXAMPLES)
    find_package(nlohmann_json 3.9.1 QUIET)
endif()

# System provided, third party libraries (not bundled!)
set(GINKGO_HAVE_HWLOC 0)
if(GINKGO_BUILD_HWLOC)
    find_package(HWLOC 2.1 REQUIRED)
    set(GINKGO_HAVE_HWLOC 1)
    message(
        WARNING
        "The GINKGO_BUILD_HWLOC option has no beneficial effect. Consider setting it to GINKGO_BUILD_HWLOC=OFF."
    )
endif()

set(GINKGO_HAVE_PAPI_SDE 0)
if(GINKGO_BUILD_PAPI_SDE)
    find_package(PAPI 7.0.1.0 COMPONENTS sde)
    if(PAPI_SDE_FOUND)
        set(GINKGO_HAVE_PAPI_SDE 1)
    else()
        message(
            WARNING
            "PAPI (SDE) could not be found. PAPI_SDE support will be disabled."
        )
        set(GINKGO_BUILD_PAPI_SDE
            OFF
            CACHE BOOL
            "PAPI_SDE support was disabled because a system package could not be found."
            FORCE
        )
    endif()
endif()

# Bundled third party libraries
add_subdirectory(third_party) # Third-party tools and libraries

if(MSVC)
    if(BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    else()
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS FALSE)
    endif()
endif()

if(GINKGO_BUILD_SYCL)
    ginkgo_extract_dpcpp_version(
        ${CMAKE_CXX_COMPILER}
        GINKGO_DPCPP_MAJOR_VERSION
        __LIBSYCL_MAJOR_VERSION
    )
    ginkgo_extract_dpcpp_version(
        ${CMAKE_CXX_COMPILER}
        GINKGO_DPCPP_MINOR_VERSION
        __LIBSYCL_MINOR_VERSION
    )
    ginkgo_extract_dpcpp_version(
        ${CMAKE_CXX_COMPILER}
        GINKGO_DPCPP_VERSION
        __SYCL_COMPILER_VERSION
    )
else()
    set(GINKGO_DPCPP_MAJOR_VERSION "0")
    set(GINKGO_DPCPP_MINOR_VERSION "0")
endif()

ginkgo_generate_ginkgo_hpp()
configure_file(
    ${Ginkgo_SOURCE_DIR}/include/ginkgo/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/config.hpp
    @ONLY
)
configure_file(
    ${Ginkgo_SOURCE_DIR}/include/ginkgo/extensions/kokkos/config.hpp.in
    ${Ginkgo_BINARY_DIR}/include/ginkgo/extensions/kokkos/config.hpp
    @ONLY
)

# Ginkgo core libraries
# Needs to be first in order for `CMAKE_CUDA_DEVICE_LINK_EXECUTABLE` to be
# propagated to the other parts of Ginkgo in case of building as static libraries
add_subdirectory(devices) # Basic device functionalities. Always compiled.
add_subdirectory(common) # Import list of unified kernel source files
if(GINKGO_BUILD_CUDA)
    add_subdirectory(cuda) # High-performance kernels for NVIDIA GPUs
endif()
if(GINKGO_BUILD_REFERENCE)
    add_subdirectory(reference) # Reference kernel implementations
endif()
if(GINKGO_BUILD_HIP)
    add_subdirectory(hip) # High-performance kernels for AMD or NVIDIA GPUs
endif()
if(GINKGO_BUILD_SYCL)
    add_subdirectory(dpcpp) # High-performance DPC++ kernels
endif()
if(GINKGO_BUILD_OMP)
    add_subdirectory(omp) # High-performance omp kernels
endif()
add_subdirectory(core) # Core Ginkgo types and top-level functions
add_subdirectory(include) # Public API self-contained check
if(GINKGO_BUILD_TESTS)
    add_subdirectory(test) # Tests running on all executors
endif()

# Non core directories and targets
add_subdirectory(extensions)

if(GINKGO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(GINKGO_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

if(GINKGO_DEVEL_TOOLS)
    find_program(PRE_COMMIT pre-commit)
    if(NOT PRE_COMMIT)
        message(
            FATAL_ERROR
            "The pre-commit command was not found. It is necessary if you want to commit changes to Ginkgo. "
            "If that is not the case, set GINKGO_DEVEL_TOOLS=OFF. "
            "Otherwise install pre-commit via pipx (or pip) using:\n"
            "    pipx install pre-commit"
        )
    endif()

    execute_process(
        COMMAND "${PRE_COMMIT}" "install"
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
        RESULT_VARIABLE pre-commit-result
        OUTPUT_VARIABLE pre-commit-output
        ERROR_VARIABLE pre-commit-error
    )
    if(pre-commit-result)
        message(
            FATAL_ERROR
            "Failed to install the git hooks via pre-commit. Please check the error message:\n"
            "${pre-commit-output}\n${pre-commit-error}"
        )
    endif()
    if(
        pre-commit-output
            MATCHES
            "^Running in migration mode with existing hooks"
    )
        message(
            WARNING
            "An existing git hook was encountered during `pre-commit install`. The old git hook "
            "will also be executed. Consider removing it with `pre-commit install -f`"
        )
    elseif(NOT pre-commit-output MATCHES "^pre-commit installed at")
        message(
            WARNING
            "`pre-commit install` did not exit normally. Please check the output message:\n"
            "${pre-commit-output}"
        )
    endif()

    add_custom_target(
        format
        COMMAND bash -c "${PRE_COMMIT} run"
        WORKING_DIRECTORY ${Ginkgo_SOURCE_DIR}
        VERBATIM
    )
endif()

# Installation
include(cmake/information_helpers.cmake)
ginkgo_pkg_information()
ginkgo_git_information()

include(cmake/get_info.cmake)

if(GINKGO_BUILD_DOC)
    add_subdirectory(doc)
endif()

# WINDOWS NVCC has " inside the string, add escape character
# to avoid config problem.
ginkgo_modify_flags(CMAKE_CUDA_FLAGS)
ginkgo_modify_flags(CMAKE_CUDA_FLAGS_DEBUG)
ginkgo_modify_flags(CMAKE_CUDA_FLAGS_RELEASE)
ginkgo_install()
ginkgo_export_binary_dir()

set(GINKGO_TEST_INSTALL_SRC_DIR "${Ginkgo_SOURCE_DIR}/test/test_install/")
set(GINKGO_TEST_INSTALL_BIN_DIR "${Ginkgo_BINARY_DIR}/test/test_install/")
set(GINKGO_TEST_EXPORTBUILD_SRC_DIR
    "${Ginkgo_SOURCE_DIR}/test/test_exportbuild/"
)
set(GINKGO_TEST_EXPORTBUILD_BIN_DIR
    "${Ginkgo_BINARY_DIR}/test/test_exportbuild/"
)
set(GINKGO_TEST_PKGCONFIG_SRC_DIR "${Ginkgo_SOURCE_DIR}/test/test_pkgconfig/")
set(GINKGO_TEST_PKGCONFIG_BIN_DIR "${Ginkgo_BINARY_DIR}/test/test_pkgconfig/")
get_property(GINKGO_USE_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
# GINKGO_CONFIG_PREFIX contains / in the end.
set(GINKGO_CONFIG_PREFIX "$<$<BOOL:${GINKGO_USE_MULTI_CONFIG}>:$<CONFIG>/>")
set(GINKGO_TEST_INSTALL_CMD
    ${GINKGO_TEST_INSTALL_BIN_DIR}/${GINKGO_CONFIG_PREFIX}test_install
)
set(GINKGO_TEST_EXPORTBUILD_CMD
    ${GINKGO_TEST_EXPORTBUILD_BIN_DIR}/${GINKGO_CONFIG_PREFIX}test_exportbuild
)
set(GINKGO_TEST_PKGCONFIG_CMD
    ${GINKGO_TEST_PKGCONFIG_BIN_DIR}/${GINKGO_CONFIG_PREFIX}test_pkgconfig
)
if(GINKGO_BUILD_CUDA)
    set(GINKGO_TEST_INSTALL_CUDA_CMD
        ${GINKGO_TEST_INSTALL_BIN_DIR}/${GINKGO_CONFIG_PREFIX}test_install_cuda
    )
endif()
if(GINKGO_BUILD_HIP)
    set(GINKGO_TEST_INSTALL_HIP_CMD
        ${GINKGO_TEST_INSTALL_BIN_DIR}/${GINKGO_CONFIG_PREFIX}test_install_hip
    )
endif()

file(MAKE_DIRECTORY "${GINKGO_TEST_INSTALL_BIN_DIR}")
file(MAKE_DIRECTORY "${GINKGO_TEST_EXPORTBUILD_BIN_DIR}")
set(TOOLSET "")
if(NOT "${CMAKE_GENERATOR_TOOLSET}" STREQUAL "")
    set(TOOLSET "-T${CMAKE_GENERATOR_TOOLSET}")
endif()
add_custom_target(
    test_install
    COMMAND
        ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} ${TOOLSET}
        -S${GINKGO_TEST_INSTALL_SRC_DIR} -B${GINKGO_TEST_INSTALL_BIN_DIR}
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
    # `--config cfg` is ignored by single-configuration generator.
    # `$<CONFIG>` is always be the same as `CMAKE_BUILD_TYPE` in
    # single-configuration generator.
    COMMAND
        ${CMAKE_COMMAND} --build ${GINKGO_TEST_INSTALL_BIN_DIR} --config
        $<CONFIG>
    COMMAND ${GINKGO_TEST_INSTALL_CMD}
    COMMAND ${GINKGO_TEST_INSTALL_CUDA_CMD}
    COMMAND ${GINKGO_TEST_INSTALL_HIP_CMD}
    WORKING_DIRECTORY ${GINKGO_TEST_INSTALL_BIN_DIR}
    COMMENT "Running a test on the installed binaries. "
    "This requires running `(sudo) make install` first."
)

add_custom_target(
    test_exportbuild
    COMMAND
        ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} ${TOOLSET}
        -S${GINKGO_TEST_EXPORTBUILD_SRC_DIR}
        -B${GINKGO_TEST_EXPORTBUILD_BIN_DIR}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} -DGinkgo_ROOT=${Ginkgo_BINARY_DIR}
    # `--config cfg` is ignored by single-configuration generator.
    # `$<CONFIG>` is always be the same as `CMAKE_BUILD_TYPE` in
    # single-configuration generator.
    COMMAND
        ${CMAKE_COMMAND} --build ${GINKGO_TEST_EXPORTBUILD_BIN_DIR} --config
        $<CONFIG>
    COMMAND ${GINKGO_TEST_EXPORTBUILD_CMD}
    COMMENT "Running a test on Ginkgo's exported build directory."
)

# static linking with pkg-config is not possible with HIP, since
# some linker information cannot be expressed in pkg-config files
if(BUILD_SHARED_LIBS OR NOT GINKGO_BUILD_HIP)
    add_custom_target(
        test_pkgconfig
        COMMAND
            ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} ${TOOLSET}
            -S${GINKGO_TEST_PKGCONFIG_SRC_DIR}
            -B${GINKGO_TEST_PKGCONFIG_BIN_DIR}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DCMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}
        # `--config cfg` is ignored by single-configuration generator.
        # `$<CONFIG>` is always be the same as `CMAKE_BUILD_TYPE` in
        # single-configuration generator.
        COMMAND
            ${CMAKE_COMMAND} --build ${GINKGO_TEST_PKGCONFIG_BIN_DIR} --config
            $<CONFIG>
        COMMAND ${GINKGO_TEST_PKGCONFIG_CMD}
        COMMENT "Running a test on Ginkgo's PkgConfig"
        "This requires installing Ginkgo first"
    )
endif()

# Setup CPack
set(CPACK_PACKAGE_DESCRIPTION_FILE "${Ginkgo_SOURCE_DIR}/README.md")
set(CPACK_RESOURCE_FILE_LICENSE "${Ginkgo_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_ICON "${Ginkgo_SOURCE_DIR}/assets/logo.png")
set(CPACK_PACKAGE_CONTACT "ginkgo.library@gmail.com")
include(CPack)

# And finally, print the configuration to screen:
if(GINKGO_CONFIG_LOG_DETAILED)
    file(READ ${PROJECT_BINARY_DIR}/detailed.log GINKGO_LOG_SUMMARY)
else()
    file(READ ${PROJECT_BINARY_DIR}/minimal.log GINKGO_LOG_SUMMARY)
endif()
message(STATUS "${GINKGO_LOG_SUMMARY}")

# make sure no build files get committed accidentally
if(NOT EXISTS ${CMAKE_CURRENT_BINARY_DIR}/.gitignore)
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.gitignore "*")
endif()
