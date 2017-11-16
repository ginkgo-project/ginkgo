if(NOT BUILD_NVIDIA)
    add_library(ginkgo_nvidia SHARED nvidia.cpp)
    target_include_directories(ginkgo_nvidia PUBLIC ${PROJECT_SOURCE_DIR})
endif()

