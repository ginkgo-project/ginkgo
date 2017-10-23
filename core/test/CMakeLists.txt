macro(create_test test_name)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE msparse gtest_main)
    add_test(NAME core/test/${test_name} COMMAND ${test_name})
endmacro(create_test)

create_test(exception)
create_test(exception_helpers)
