add_executable(ginkgo-ranges ginkgo-ranges.cpp)
target_link_libraries(ginkgo-ranges ginkgo)
target_include_directories(ginkgo-ranges PRIVATE ${PROJECT_SOURCE_DIR})
