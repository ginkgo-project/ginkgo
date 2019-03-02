if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginko is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
endif()

add_subdirectory(matrix_generator)
add_subdirectory(matrix_statistics)
add_subdirectory(preconditioner)
add_subdirectory(solver)
add_subdirectory(spmv)
add_subdirectory(spmv_comparison_cuda)

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
