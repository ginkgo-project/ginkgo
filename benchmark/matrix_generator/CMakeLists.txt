add_executable(matrix_generator matrix_generator.cpp)
target_link_libraries(matrix_generator ginkgo gflags)
target_include_directories(matrix_generator PRIVATE
    ${Ginkgo_BINARY_DIR}/third_party/rapidjson/src/include)
