if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginkgo is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
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
