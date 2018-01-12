set(SOURCES
    matrix/dense_kernels.cpp)

add_library(ginkgo_reference SHARED
    $<TARGET_OBJECTS:ginkgo_reference_device>
    ${SOURCES})
target_include_directories(ginkgo_reference PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()
