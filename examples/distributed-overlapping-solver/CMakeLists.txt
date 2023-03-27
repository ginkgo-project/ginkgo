add_executable(distributed-overlapping-solver distributed-overlapping-solver.cpp)
target_link_libraries(distributed-overlapping-solver Ginkgo::ginkgo)

add_executable(distributed-overlapping-solver-2d distributed-overlapping-solver-2d.cpp)
target_link_libraries(distributed-overlapping-solver-2d Ginkgo::ginkgo)

add_executable(non-overlapping-solver-2d non-overlapping-solver-2d.cpp)
target_link_libraries(non-overlapping-solver-2d Ginkgo::ginkgo)
