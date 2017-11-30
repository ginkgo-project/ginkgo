find_package(CUDA REQUIRED)
add_library(ginkgo_gpu SHARED
    $<TARGET_OBJECTS:ginkgo_gpu_device>
    executor.cpp exception.hpp)
target_include_directories(ginkgo_gpu PUBLIC ${PROJECT_SOURCE_DIR})
target_include_directories(ginkgo_gpu
    SYSTEM PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

target_link_libraries(ginkgo_gpu ${CUDA_LIBRARIES})
