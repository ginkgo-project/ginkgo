set(SOURCES
    base/executor.cpp)

add_subdirectory(devices)  # basic device functionalities, always compiled
add_subdirectory(device_hooks)  # placeholders for disabled modules

add_library(ginkgo SHARED ${SOURCES})
target_include_directories(ginkgo PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(ginkgo PUBLIC ginkgo_cpu ginkgo_gpu ginkgo_reference)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

