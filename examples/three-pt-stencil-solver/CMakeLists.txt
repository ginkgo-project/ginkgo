add_executable(three-pt-stencil-solver three-pt-stencil-solver.cpp)
target_link_libraries(three-pt-stencil-solver ginkgo)
target_include_directories(three-pt-stencil-solver PRIVATE ${PROJECT_SOURCE_DIR})
