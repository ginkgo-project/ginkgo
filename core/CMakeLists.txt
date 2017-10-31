set(SOURCES
    base/executor.cpp)

add_subdirectory(device_hooks)  # placeholders for disabled modules

add_library(msparse SHARED ${SOURCES})
target_include_directories(msparse PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(msparse PUBLIC msparse_nvidia)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

