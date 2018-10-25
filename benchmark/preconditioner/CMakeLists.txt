add_executable(preconditioner preconditioner.cpp)
target_link_libraries(preconditioner ginkgo gflags)
target_include_directories(preconditioner PRIVATE
    ${Ginkgo_BINARY_DIR}/third_party/rapidjson/src/include)
