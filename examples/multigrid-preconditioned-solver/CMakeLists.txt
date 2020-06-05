add_executable(multigrid-preconditioned-solver multigrid-preconditioned-solver.cpp)
target_link_libraries(multigrid-preconditioned-solver ginkgo)
target_include_directories(multigrid-preconditioned-solver PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
