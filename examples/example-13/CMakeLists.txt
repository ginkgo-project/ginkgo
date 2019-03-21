add_executable(ranges ranges.cpp)
target_link_libraries(ranges ginkgo)
target_include_directories(ranges PRIVATE ${PROJECT_SOURCE_DIR})
