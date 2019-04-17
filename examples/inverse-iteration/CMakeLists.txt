add_executable(inverse-iteration inverse-iteration.cpp)
target_link_libraries(inverse-iteration ginkgo)
target_include_directories(inverse-iteration PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
