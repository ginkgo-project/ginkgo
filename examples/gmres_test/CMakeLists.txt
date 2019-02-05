enable_language(CUDA)

find_library(CUDA_RUNTIME_LIBS cudart
    HINT ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

add_executable(gmres_test gmres_test.cpp)
target_link_libraries(gmres_test ginkgo ${CUDA_RUNTIME_LIBS})
target_include_directories(gmres_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${PROJECT_SOURCE_DIR})
