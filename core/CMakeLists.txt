set(SOURCES
    executor.cpp)

add_library(msparse SHARED ${SOURCES})
target_include_directories(msparse PUBLIC .)

if(BUILD_TESTS)
    add_subdirectory(test)
endif()

