set(SOURCES
    base/executor.cpp)

add_library(msparse SHARED ${SOURCES})
target_include_directories(msparse PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

