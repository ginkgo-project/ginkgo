include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)
FetchContent_MakeAvailable(googletest)

include(GoogleTest)

function(accessor_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(
        ${name}
        PRIVATE MemoryAccessor::accessor GTest::gtest_main
    )
    gtest_discover_tests(${name})
endfunction()

accessor_test(index_span)
accessor_test(range)
accessor_test(math)
accessor_test(block_col_major)
accessor_test(row_major)
accessor_test(reduced_row_major)
accessor_test(reduced_row_major_reference)
accessor_test(scaled_reduced_row_major)
accessor_test(scaled_reduced_row_major_reference)
accessor_test(sycl_helper)
