add_executable(example-8 example-8.cpp)
target_link_libraries(example-8 ginkgo)
target_include_directories(example-8 PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
