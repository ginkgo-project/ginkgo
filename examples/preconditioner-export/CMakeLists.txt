add_executable(preconditioner-export preconditioner-export.cpp)
target_link_libraries(preconditioner-export ginkgo)
target_include_directories(preconditioner-export PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
