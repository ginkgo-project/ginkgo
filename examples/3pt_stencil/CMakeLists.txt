add_executable(3pt_stencil 3pt_stencil.cpp)
target_link_libraries(3pt_stencil ginkgo)
target_include_directories(3pt_stencil PRIVATE ${PROJECT_SOURCE_DIR})
