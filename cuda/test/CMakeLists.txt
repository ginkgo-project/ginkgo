include(${PROJECT_SOURCE_DIR}/cmake/create_test.cmake)

add_subdirectory(base)
add_subdirectory(components)
add_subdirectory(solver)
add_subdirectory(utils)
add_executable(reproducer test.cu)
add_test(NAME reproducer COMMAND reproducer)