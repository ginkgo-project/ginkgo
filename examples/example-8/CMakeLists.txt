add_executable(inverse_iteration inverse_iteration.cpp)
target_link_libraries(inverse_iteration ginkgo)
target_include_directories(inverse_iteration PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
