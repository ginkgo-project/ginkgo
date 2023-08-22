if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginkgo is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
endif()

function(ginkgo_benchmark_add_tuning_maybe name)
    if(GINKGO_BENCHMARK_ENABLE_TUNING)
        target_sources(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../utils/tuning_variables.cpp)
    endif()
endfunction()

function(ginkgo_benchmark_cusparse_linops type def)
    add_library(cusparse_linops_${type} utils/cuda_linops.cpp)
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        # remove false positive CUDA warnings when calling one<T>() and zero<T>()
        target_compile_options(cusparse_linops_${type}
            PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    endif()
    # make the dependency public to catch issues
    target_compile_definitions(cusparse_linops_${type} PUBLIC ${def})
    target_link_libraries(cusparse_linops_${type} Ginkgo::ginkgo CUDA::cudart CUDA::cublas CUDA::cusparse)
endfunction()

function(ginkgo_benchmark_hipsparse_linops type def)
    add_library(hipsparse_linops_${type} utils/hip_linops.hip.cpp)
    target_compile_definitions(hipsparse_linops_${type} PUBLIC ${def})
    EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
    set_target_properties(hipsparse_linops_${type} PROPERTIES COMPILE_FLAGS ${HIP_CXX_FLAGS})
    # use Thrust C++ device just for compilation, we don't use thrust::complex in the benchmarks
    target_compile_definitions(hipsparse_linops_${type} PUBLIC -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP)
    target_include_directories(hipsparse_linops_${type} SYSTEM PRIVATE
        ${HSA_HEADER} ${HIP_INCLUDE_DIRS}
        ${HIPBLAS_INCLUDE_DIRS} ${HIPSPARSE_INCLUDE_DIRS})
    target_link_libraries(hipsparse_linops_${type} Ginkgo::ginkgo ${HIPSPARSE_LIBRARIES})
endfunction()

function(ginkgo_benchmark_onemkl_linops type def)
    add_library(onemkl_linops_${type} utils/dpcpp_linops.dp.cpp)
    # make the dependency public to catch issues
    target_compile_definitions(onemkl_linops_${type} PUBLIC ${def})
    target_compile_features(onemkl_linops_${type} PRIVATE cxx_std_17)
    target_link_libraries(onemkl_linops_${type} PRIVATE Ginkgo::ginkgo MKL::MKL_DPCPP)
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
function(ginkgo_add_single_benchmark_executable name use_lib_linops macro_def type)
    add_executable("${name}" ${ARGN})
    target_link_libraries("${name}" ginkgo gflags rapidjson)
    # always include the device timer
    if (GINKGO_BUILD_CUDA)
        target_compile_definitions("${name}" PRIVATE HAS_CUDA_TIMER=1)
        target_link_libraries("${name}" cuda_timer)
    endif()
    if (GINKGO_BUILD_HIP)
        target_compile_definitions("${name}" PRIVATE HAS_HIP_TIMER=1)
        target_link_libraries("${name}" hip_timer)
    endif()
    if (GINKGO_BUILD_SYCL)
        target_compile_definitions("${name}" PRIVATE HAS_SYCL_TIMER=1)
        target_link_libraries("${name}" dpcpp_timer)
    endif()
    if (GINKGO_BUILD_MPI)
        target_compile_definitions(${name} PRIVATE HAS_MPI_TIMER=1)
        target_link_libraries(${name} mpi_timer)
    endif()
    target_compile_definitions("${name}" PRIVATE "${macro_def}")
    target_compile_options("${name}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    ginkgo_benchmark_add_tuning_maybe("${name}")
    if("${use_lib_linops}")
        if (GINKGO_BUILD_CUDA)
            target_compile_definitions("${name}" PRIVATE HAS_CUDA=1)
            target_link_libraries("${name}" cusparse_linops_${type})
        endif()
        if (GINKGO_BUILD_HIP)
            target_compile_definitions("${name}" PRIVATE HAS_HIP=1)
            target_link_libraries("${name}" hipsparse_linops_${type})
        endif()
        if (GINKGO_BUILD_SYCL)
            target_compile_definitions("${name}" PRIVATE HAS_SYCL=1)
            target_link_libraries("${name}" onemkl_linops_${type})
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
        "${name}" "${use_lib_linops}" "GKO_BENCHMARK_USE_DOUBLE_PRECISION" "d" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_single" "${use_lib_linops}" "GKO_BENCHMARK_USE_SINGLE_PRECISION" "s" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_dcomplex" "${use_lib_linops}" "GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION" "z" ${ARGN})
    ginkgo_add_single_benchmark_executable(
        "${name}_scomplex" "${use_lib_linops}" "GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION" "c" ${ARGN})
endfunction(ginkgo_add_typed_benchmark_executables)


if (GINKGO_BUILD_CUDA)
    ginkgo_benchmark_cusparse_linops(d GKO_BENCHMARK_USE_DOUBLE_PRECISION)
    ginkgo_benchmark_cusparse_linops(s GKO_BENCHMARK_USE_SINGLE_PRECISION)
    ginkgo_benchmark_cusparse_linops(z GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION)
    ginkgo_benchmark_cusparse_linops(c GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION)
    add_library(cuda_timer utils/cuda_timer.cpp)
    target_link_libraries(cuda_timer ginkgo CUDA::cudart)
endif()
if (GINKGO_BUILD_HIP)
    ginkgo_benchmark_hipsparse_linops(d GKO_BENCHMARK_USE_DOUBLE_PRECISION)
    ginkgo_benchmark_hipsparse_linops(s GKO_BENCHMARK_USE_SINGLE_PRECISION)
    ginkgo_benchmark_hipsparse_linops(z GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION)
    ginkgo_benchmark_hipsparse_linops(c GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION)
    add_library(hip_timer utils/hip_timer.hip.cpp)
    EXECUTE_PROCESS(COMMAND ${HIP_PATH}/bin/hipconfig --cpp_config OUTPUT_VARIABLE HIP_CXX_FLAGS)
    set_target_properties(hip_timer PROPERTIES COMPILE_FLAGS ${HIP_CXX_FLAGS})
    target_include_directories(hip_timer SYSTEM PRIVATE ${HSA_HEADER} ${HIP_INCLUDE_DIRS})
    target_link_libraries(hip_timer ginkgo)
endif()

if (GINKGO_BUILD_SYCL)
    ginkgo_benchmark_onemkl_linops(d GKO_BENCHMARK_USE_DOUBLE_PRECISION)
    ginkgo_benchmark_onemkl_linops(s GKO_BENCHMARK_USE_SINGLE_PRECISION)
    ginkgo_benchmark_onemkl_linops(z GKO_BENCHMARK_USE_DOUBLE_COMPLEX_PRECISION)
    ginkgo_benchmark_onemkl_linops(c GKO_BENCHMARK_USE_SINGLE_COMPLEX_PRECISION)
    add_library(dpcpp_timer utils/dpcpp_timer.dp.cpp)
    target_compile_options(dpcpp_timer PRIVATE ${GINKGO_SYCL_FLAGS})
    gko_add_sycl_to_target(TARGET dpcpp_timer SOURCES utils/dpcpp_timer.dp.cpp)
    target_link_libraries(dpcpp_timer ginkgo)
endif()

if (GINKGO_BUILD_MPI)
    add_library(mpi_timer ${Ginkgo_SOURCE_DIR}/benchmark/utils/mpi_timer.cpp)
    target_link_libraries(mpi_timer ginkgo)
endif()

add_subdirectory(blas)
add_subdirectory(conversions)
add_subdirectory(matrix_generator)
add_subdirectory(matrix_statistics)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(sparse_blas)
add_subdirectory(spmv)
add_subdirectory(tools)
if (GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()

configure_file(run_all_benchmarks.sh run_all_benchmarks.sh COPYONLY)

add_custom_target(benchmark)
add_custom_command(
    TARGET benchmark POST_BUILD
    COMMAND bash run_all_benchmarks.sh >/dev/null
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
