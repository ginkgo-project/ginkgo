if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginkgo is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
endif()

if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_HIP AND
    GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
    message(FATAL_ERROR "Building the benchmarks for both HIP AMD and CUDA "
        "at the same time is currently not supported. "
        "Disable the benchmark build using `-DGINKGO_BUILD_BENCHMARKS=OFF` "
        "or use `export HIP_PLATFORM=nvcc` (ROCM <=4.0) or "
        "`export HIP_PLATFORM=nvidia` (ROCM >= 4.1) in your build environment instead.")
endif()

function(ginkgo_benchmark_add_tuning_maybe name)
    if(GINKGO_BENCHMARK_ENABLE_TUNING)
        target_sources(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils/tuning_variables.cpp)
    endif()
endfunction()

function(ginkgo_benchmark_cusp_linops name)
    target_compile_definitions("${name}" PRIVATE HAS_CUDA=1)
    target_link_libraries("${name}" ginkgo ${CUDA_RUNTIME_LIBS}
        ${CUBLAS} ${CUSPARSE})
    target_include_directories("${name}" SYSTEM PRIVATE ${CUDA_INCLUDE_DIRS})
    if(CMAKE_CUDA_COMPILER_VERSION GREATER_EQUAL "9.2")
        target_compile_definitions("${name}" PRIVATE ALLOWMP=1)
    endif()
endfunction()

function(ginkgo_benchmark_hipsp_linops name)
    target_compile_definitions("${name}" PRIVATE HAS_HIP=1)
    EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
    set_target_properties("${name}" PROPERTIES COMPILE_FLAGS ${HIP_CXX_FLAGS})
    # use Thrust C++ device just for compilation, we don't use thrust::complex in the benchmarks
    target_compile_definitions("${name}" PUBLIC -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    # for some reason, HIP creates a dependency on Threads::Threads here, so we
    # need to find it
    find_package(Threads REQUIRED)
    find_package(HIP REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hiprand REQUIRED)
    find_package(hipsparse REQUIRED)
    find_package(rocrand REQUIRED)
    target_include_directories("${name}" SYSTEM PRIVATE
        ${HSA_HEADER} ${HIP_INCLUDE_DIRS}
        ${HIPBLAS_INCLUDE_DIRS} ${HIPSPARSE_INCLUDE_DIRS})

    target_link_libraries("${name}" ${HIPSPARSE_LIBRARIES})
endfunction()


# Generates an executable for one precision. Each executable will be linked to
# `ginkgo`, `gflags` and `rapidjson`.
# Note: This should only be used by `ginkgo_add_typed_benchmark_executables`
#
# \param name            name for the executable to create (including type suffix)
# \param use_lib_linops  Boolean indicating if linking against hipsparse/cusparse
#                        is necessary
# \param macro_def       preprocessor macro name that will be defined during
#                        building (to compile for a specific type)
# All remaining arguments will be treated as source files
function(ginkgo_add_single_benchmark_executable name use_lib_linops macro_def)
    add_executable("${name}" ${ARGN})
    target_link_libraries("${name}" ginkgo gflags rapidjson)
    target_compile_definitions("${name}" PRIVATE "${macro_def}")
    ginkgo_benchmark_add_tuning_maybe("${name}")
    if("${use_lib_linops}")
        if (GINKGO_BUILD_CUDA)
            ginkgo_benchmark_cusp_linops("${name}")
        endif()
        if (GINKGO_BUILD_HIP)
            ginkgo_benchmark_hipsp_linops("${name}")
        endif()
    endif()
endfunction(ginkgo_add_single_benchmark_executable)


# Generates an executable for each supported precision. Each executable will be
# linked to `ginkgo`, `gflags` and `rapidjson`.
#
# \param name            base-name for the executable to create
# \param use_lib_linops  Boolean indicating if linking against hipsparse/cusparse
#                        is necessary
# All remaining arguments will be treated as source files
function(ginkgo_add_typed_benchmark_executables name use_lib_linops)
    ginkgo_add_single_benchmark_executable(
        "${name}" "${use_lib_linops}" "GKO_BENCHMARK_USE_DOUBLE_PRECISION" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_single" "${use_lib_linops}" "GKO_BENCHMARK_USE_SINGLE_PRECISION" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_dcomplex" "${use_lib_linops}" "GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_scomplex" "${use_lib_linops}" "GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION" ${ARGN})
endfunction(ginkgo_add_typed_benchmark_executables)


add_subdirectory(conversions)
add_subdirectory(matrix_generator)
add_subdirectory(matrix_statistics)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(spmv)

add_custom_target(make_run_all_benchmarks ALL)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/run_all_benchmarks.sh
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

add_custom_command(
    TARGET make_run_all_benchmarks POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/run_all_benchmarks.sh
            ${CMAKE_CURRENT_BINARY_DIR}/run_all_benchmarks.sh)

add_custom_target(benchmark)
add_custom_command(
    TARGET benchmark POST_BUILD
    COMMAND bash run_all_benchmarks.sh >/dev/null
    DEPENDS make_run_all_benchmarks
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
