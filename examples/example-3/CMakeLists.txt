add_executable(poisson_solver poisson_solver.cpp)
target_link_libraries(poisson_solver ginkgo)
target_include_directories(poisson_solver PRIVATE ${PROJECT_SOURCE_DIR})
