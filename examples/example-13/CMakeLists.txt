add_executable(example-13 example-13.cpp)
target_link_libraries(example-13 ginkgo)
target_include_directories(example-13 PRIVATE ${PROJECT_SOURCE_DIR})
