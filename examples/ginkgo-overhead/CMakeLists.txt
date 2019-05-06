add_executable(ginkgo-overhead ginkgo-overhead.cpp)
target_link_libraries(ginkgo-overhead ginkgo)
target_include_directories(ginkgo-overhead PRIVATE ${PROJECT_SOURCE_DIR})
