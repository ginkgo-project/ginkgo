add_executable(example-3 example-3.cpp)
target_link_libraries(example-3 ginkgo)
target_include_directories(example-3 PRIVATE ${PROJECT_SOURCE_DIR})
