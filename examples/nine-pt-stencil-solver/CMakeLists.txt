add_executable(nine-pt-stencil-solver nine-pt-stencil-solver.cpp)
target_link_libraries(nine-pt-stencil-solver ginkgo)
target_include_directories(nine-pt-stencil-solver PRIVATE ${PROJECT_SOURCE_DIR})
