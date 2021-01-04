set(target_name "ginkgo-overhead")
add_executable(${target_name} ${target_name}.cpp)
target_link_libraries(${target_name} Ginkgo::ginkgo)
target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
