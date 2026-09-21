add_executable(distributed-poisson-matrix-free-split distributed-poisson-matrix-free-split.cpp)
target_link_libraries(distributed-poisson-matrix-free-split Ginkgo::ginkgo)