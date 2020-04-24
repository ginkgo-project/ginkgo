add_executable(adaptiveprecision-blockjacobi adaptiveprecision-blockjacobi.cpp)
target_link_libraries(adaptiveprecision-blockjacobi ginkgo)
target_include_directories(adaptiveprecision-blockjacobi PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
