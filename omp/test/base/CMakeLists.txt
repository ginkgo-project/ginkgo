ginkgo_create_omp_test(kernel_launch)
ginkgo_create_omp_test(index_set)

# Rerun the reduction tests with fewer threads granted than requested, without
# nesting: OMP_NUM_THREADS=8 makes the kernels request up to 8 threads, while
# OMP_THREAD_LIMIT=4 caps the team at 4. The *UndersizedTeam tests only
# cover the team-of-1 case.
# KMP_WARNINGS=0: older libomp (e.g. LLVM 8) aborts with "invalid %N$ use
# detected" when printing its "cannot form a team" warning.
add_test(
    NAME omp/test/base/kernel_launch_thread_limited
    COMMAND omp_test_base_kernel_launch --gtest_filter=*Reduction*
    WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>"
)
set_tests_properties(
    omp/test/base/kernel_launch_thread_limited
    PROPERTIES
        ENVIRONMENT
            "OMP_NUM_THREADS=8;OMP_THREAD_LIMIT=4;OMP_MAX_ACTIVE_LEVELS=1;KMP_WARNINGS=0"
)

if(GINKGO_OMP_WITH_HIGHWAY)
    # Idempotent, and repeated here on purpose: this directory must not depend
    # on omp/CMakeLists.txt having included highway.cmake first.
    include(${PROJECT_SOURCE_DIR}/cmake/highway.cmake)
    # reduce_sum_op/reduce_max_op gain members only under GKO_OMP_HIGHWAY, so
    # this target must define it too or the two TUs disagree on the class.
    gko_add_highway(omp_test_base_kernel_launch)
    ginkgo_create_omp_test(kernel_launch_simd)
    gko_add_highway(omp_test_base_kernel_launch_simd)
    # The bit-identity tests compare a Highway expression against a scalar loop;
    # contraction would fuse the scalar side on some targets only. Shapes the
    # layer fuses deliberately are compared against std::fma instead.
    # MSVC does not recognise the option.
    target_compile_options(
        omp_test_base_kernel_launch_simd
        PRIVATE
            $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang,IntelLLVM,NVHPC>:-ffp-contract=off>
    )
endif()
