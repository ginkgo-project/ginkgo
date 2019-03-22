add_executable(example-5 example-5.cpp)
target_link_libraries(example-5 ginkgo)
target_include_directories(example-5 PRIVATE ${PROJECT_SOURCE_DIR})
