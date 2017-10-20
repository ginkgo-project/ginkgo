set(CORE_SOURCES
    executor.cpp)

add_library(msparse_core OBJECT ${CORE_SOURCES})
target_include_directories(msparse_core PUBLIC .)

if (BUILD_TESTS)
    add_subdirectory(test)
endif()

