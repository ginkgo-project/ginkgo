add_executable(simple_spmv simple_spmv.cpp)
target_link_libraries(simple_spmv ginkgo gtest_main)
target_include_directories(simple_spmv PRIVATE ${PROJECT_SOURCE_DIR})
