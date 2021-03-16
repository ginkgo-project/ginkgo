# Creates a test that only adds `Ginkgo_SOURCE_DIR` to the include directories
# to make sure the accessors are truly independent of Ginkgo and can be used
# as a header-only library.
function(create_accessor_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable("${TEST_TARGET_NAME}" "${test_name}.cpp")
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "${Ginkgo_SOURCE_DIR}"
        )
    set_target_properties("${TEST_TARGET_NAME}" PROPERTIES
        OUTPUT_NAME "${test_name}")
    target_link_libraries("${TEST_TARGET_NAME}" PRIVATE GTest::Main GTest::GTest ${ARGN})
    add_test(NAME "${REL_BINARY_DIR}/${test_name}" COMMAND "${TEST_TARGET_NAME}")
endfunction(create_accessor_test)

create_accessor_test(index_span)
create_accessor_test(range)
# Accessor tests:
create_accessor_test(block_col_major)
create_accessor_test(reduced_row_major)
ginkgo_create_test(reduced_row_major_ginkgo) # needs Ginkgo for gko::half
create_accessor_test(row_major)
create_accessor_test(scaled_reduced_row_major)
