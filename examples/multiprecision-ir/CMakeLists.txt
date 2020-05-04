add_executable(multiprecision-ir multiprecision-ir.cpp)
target_link_libraries(multiprecision-ir ginkgo)
target_include_directories(multiprecision-ir PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
