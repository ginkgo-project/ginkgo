add_executable(expression-templates expression-templates.cpp)
target_link_libraries(expression-templates ginkgo)
target_include_directories(expression-templates PRIVATE ${PROJECT_SOURCE_DIR})
