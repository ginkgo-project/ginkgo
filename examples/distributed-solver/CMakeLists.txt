add_executable(distributed-solver distributed-solver.cpp)
target_link_libraries(distributed-solver Ginkgo::ginkgo)

add_executable(distributed-solver-1d distributed-solver-1d.cpp)
target_link_libraries(distributed-solver-1d Ginkgo::ginkgo)

configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(data/b.mtx data/b.mtx COPYONLY)
configure_file(data/x.mtx data/x.mtx COPYONLY)
configure_file(data/x0.mtx data/x0.mtx COPYONLY)
