cmake_minimum_required(VERSION 3.16)
project(MemoryAccessor VERSION 1.0.0 LANGUAGES CXX)

add_library(memory_accessor INTERFACE)
add_library(MemoryAccessor::accessor ALIAS memory_accessor)

target_compile_features(memory_accessor INTERFACE cxx_std_17)

target_include_directories(
    memory_accessor
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# The accessor owns its unit tests and builds them only when configured as the
# top-level project (i.e. the standalone repo) - never when a parent such as
# Ginkgo pulls it in via add_subdirectory/FetchContent. They use their own GTest.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
    add_subdirectory(test)
endif()

# TODO TAE: (accessor-export): move this line out of here once the accessor is its own
# GitHub repo. It belongs to Ginkgo (it references Ginkgo's `Ginkgo` export set,
# which the standalone accessor repo must not know about), so it moves into the
# FetchContent wrapper ginkgo/third_party/accessor/CMakeLists.txt, right after
# the fetched `memory_accessor` target becomes available.
install(TARGETS memory_accessor EXPORT Ginkgo)
