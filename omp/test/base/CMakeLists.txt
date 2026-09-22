ginkgo_create_omp_test(kernel_launch)
ginkgo_create_omp_test(index_set)

# Rerun the reduction tests with fewer threads granted than requested, without
# nesting: OMP_NUM_THREADS=8 makes the kernels request up to 8 threads, while
# OMP_THREAD_LIMIT=4 caps the team at 4. The *UndersizedTeam tests only
# cover the team-of-1 case.
add_test(
    NAME omp/test/base/kernel_launch_thread_limited
    COMMAND omp_test_base_kernel_launch --gtest_filter=*Reduction*
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
)
set_tests_properties(
    omp/test/base/kernel_launch_thread_limited
    PROPERTIES
        ENVIRONMENT
            "OMP_NUM_THREADS=8;OMP_THREAD_LIMIT=4;OMP_MAX_ACTIVE_LEVELS=1"
)
