if(NOT BUILD_NVIDIA)
    add_library(msparse_nvidia SHARED nvidia.cpp)
    target_include_directories(msparse_nvidia PUBLIC ${PROJECT_SOURCE_DIR})
endif()

