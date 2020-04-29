add_executable(iterative-refinement iterative-refinement.cpp)
target_link_libraries(iterative-refinement ginkgo)
target_include_directories(iterative-refinement PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
