add_executable(distributed-spmv-scaling distributed-spmv-scaling.cpp)
target_link_libraries(distributed-spmv-scaling Ginkgo::ginkgo)
