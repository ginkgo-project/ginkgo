add_executable(ir-ilu-preconditioned-solver ir-ilu-preconditioned-solver.cpp)
target_link_libraries(ir-ilu-preconditioned-solver ginkgo)
target_include_directories(ir-ilu-preconditioned-solver PRIVATE ${PROJECT_SOURCE_DIR})
configure_file(data/A.mtx data/A.mtx COPYONLY)
