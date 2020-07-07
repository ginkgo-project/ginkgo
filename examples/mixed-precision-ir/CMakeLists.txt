add_executable(mixed-precision-ir mixed-precision-ir.cpp)
target_link_libraries(mixed-precision-ir ginkgo)
target_include_directories(mixed-precision-ir PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
