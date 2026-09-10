ginkgo_create_omp_test(kernel_launch)
ginkgo_create_omp_test(index_set)

# Same reduction tests under a capped team; nesting only ever gives a team of 1.
add_test(
    NAME omp/test/base/kernel_launch_thread_limited
    COMMAND omp_test_base_kernel_launch --gtest_filter=*Reduction*
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
)
set_tests_properties(
    omp/test/base/kernel_launch_thread_limited
    PROPERTIES ENVIRONMENT "OMP_THREAD_LIMIT=4;OMP_MAX_ACTIVE_LEVELS=1"
)
