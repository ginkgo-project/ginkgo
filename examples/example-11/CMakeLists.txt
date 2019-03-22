add_executable(example-11 example-11.cpp)
target_link_libraries(example-11 ginkgo)
target_include_directories(example-11 PRIVATE ${PROJECT_SOURCE_DIR})
