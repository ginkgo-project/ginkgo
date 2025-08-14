add_executable(conv conv.cpp)
target_link_libraries(conv PRIVATE Ginkgo::ginkgo)
target_include_directories(conv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
