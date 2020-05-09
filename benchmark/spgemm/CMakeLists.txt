add_executable(spgemm spgemm.cpp)
target_link_libraries(spgemm ginkgo gflags rapidjson)