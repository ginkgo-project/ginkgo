add_executable(performance-debugging performance-debugging.cpp)
target_link_libraries(performance-debugging ginkgo)
target_include_directories(performance-debugging PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
