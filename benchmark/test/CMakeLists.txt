find_package(Python3 COMPONENTS Interpreter REQUIRED)
function(add_benchmark_test test_name)
    configure_file(${test_name}.py ${test_name}.py COPYONLY)
    add_test(NAME benchmark_${test_name}
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.py $<TARGET_FILE:${test_name}>
             WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    set(regenerate_target benchmark_test_${test_name}_regenerate)
    add_custom_target(${regenerate_target}
                      COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.py $<TARGET_FILE:${test_name}> --generate
                      COMMENT "Regenerating reference output for ${test_name}"
                      WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    add_dependencies(${regenerate_target} ${test_name})
    add_dependencies(benchmark_test_regenerate ${regenerate_target})
endfunction()
add_custom_target(benchmark_test_regenerate)
configure_file(test_framework.py.in test_framework.py @ONLY)
add_benchmark_test(blas)
add_benchmark_test(conversion)
add_benchmark_test(matrix_statistics)
add_benchmark_test(preconditioner)
add_benchmark_test(solver)
add_benchmark_test(sparse_blas)
add_benchmark_test(spmv)
if (GINKGO_BUILD_MPI)
    add_benchmark_test(multi_vector_distributed)
    add_benchmark_test(spmv_distributed)
    add_benchmark_test(solver_distributed)
endif()