add_executable(exception exception.cpp)
target_link_libraries(exception PRIVATE msparse gtest_main)
add_test(NAME core/test/exception COMMAND exception)

add_executable(exception_helpers exception_helpers.cpp)
target_link_libraries(exception_helpers PRIVATE msparse gtest_main)
add_test(NAME core/test/exception_helpers COMMAND exception_helpers)

