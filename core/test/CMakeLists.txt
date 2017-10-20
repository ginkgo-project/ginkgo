add_executable(exception exception.cpp)
target_link_libraries(exception PRIVATE msparse gtest_main)
add_test(NAME core/test/exception COMMAND exception)

