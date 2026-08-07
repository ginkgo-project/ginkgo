add_executable(distributed-poisson-matrix-free-monolithic distributed-poisson-matrix-free-monolithic.cpp)
target_link_libraries(distributed-poisson-matrix-free-monolithic Ginkgo::ginkgo)