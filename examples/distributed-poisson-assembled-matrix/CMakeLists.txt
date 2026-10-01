add_executable(distributed-poisson-assembled-matrix distributed-poisson-assembled-matrix.cpp)
target_link_libraries(distributed-poisson-assembled-matrix Ginkgo::ginkgo)