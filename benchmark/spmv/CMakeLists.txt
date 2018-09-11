add_executable(spmv spmv.cpp)
target_link_libraries(spmv ginkgo gflags)
target_include_directories(spmv PRIVATE
    ${Ginkgo_BINARY_DIR}/third_party/rapidjson/src/include)
