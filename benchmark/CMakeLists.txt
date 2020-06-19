if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginkgo is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
endif()

if (GINKGO_BUILD_CUDA AND GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM MATCHES "hcc")
    message(FATAL_ERROR "Building the benchmarks for both HIP AMD and CUDA "
        "at the same time is currently not supported. "
        "Disable the benchmark build using `-DGINKGO_BUILD_BENCHMARKS=OFF` "
        "or use `export HIP_PLATFORM=nvcc` in your build environment instead.")
endif()

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
    # for some reason, HIP creates a dependency on Threads::Threads here, so we
    # need to find it
    find_package(Threads REQUIRED)
    find_package(HIP REQUIRED)
    find_package(hipblas REQUIRED)
    find_package(hipsparse REQUIRED)
    target_include_directories("${name}" SYSTEM PRIVATE
        ${HSA_HEADER} ${HIP_INCLUDE_DIRS}
        ${HIPBLAS_INCLUDE_DIRS} ${HIPSPARSE_INCLUDE_DIRS})

    if(GINKGO_HIP_PLATFORM MATCHES "hcc")
        ginkgo_hip_ban_link_hcflag(hcc::hccrt)
        ginkgo_hip_ban_link_hcflag(hcc::hc_am)
        ginkgo_hip_ban_link_hcflag(hcc::mcwamp)
        ginkgo_hip_ban_compile_hcflag(hcc::hccrt)
        ginkgo_hip_ban_compile_hcflag(hcc::hc_am)
        ginkgo_hip_ban_compile_hcflag(hcc::mcwamp)
        ginkgo_hip_clang_ban_hip_device_flags()
        target_link_libraries("${name}" hip::device)
    else()
        target_link_libraries("${name}" ${HIP_CUDA_LIBRARIES})
    endif()
    target_link_libraries("${name}" ${HIPSPARSE_LIBRARIES})
endfunction()

add_subdirectory(conversions)
add_subdirectory(matrix_generator)
add_subdirectory(matrix_statistics)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(spmv)

add_custom_target(make_run_all_benchmarks ALL)
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
