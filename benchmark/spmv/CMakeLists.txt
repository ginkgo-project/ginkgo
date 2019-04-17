add_executable(spmv spmv.cpp)
target_link_libraries(spmv ginkgo gflags rapidjson)
