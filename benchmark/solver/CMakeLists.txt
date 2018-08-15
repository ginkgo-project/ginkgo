add_executable(solver solver.cpp)
target_link_libraries(solver ginkgo gflags)
target_include_directories(solver PRIVATE
    ${Ginkgo_BINARY_DIR}/third_party/rapidjson/src/include)
