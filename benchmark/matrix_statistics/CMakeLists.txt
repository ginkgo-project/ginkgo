add_executable(matrix_statistics matrix_statistics.cpp)
target_link_libraries(matrix_statistics ginkgo gflags)
target_include_directories(matrix_statistics PRIVATE
    ${Ginkgo_BINARY_DIR}/third_party/rapidjson/src/include)
