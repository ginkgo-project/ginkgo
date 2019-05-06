add_executable(poisson-solver poisson-solver.cpp)
target_link_libraries(poisson-solver ginkgo)
target_include_directories(poisson-solver PRIVATE ${PROJECT_SOURCE_DIR})
