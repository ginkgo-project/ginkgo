add_executable(simple_solver simple_solver.cpp)
target_link_libraries(ginkgo)
target_include_directories(simple_solver PRIVATE ${PROJECT_SOURCE_DIR})
