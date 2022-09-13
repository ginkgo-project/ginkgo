pybind11_add_module(pygko
  ${PROJECT_SOURCE_DIR}/bindings/python.cpp
  ${PROJECT_SOURCE_DIR}/bindings/dense.cpp
  ${PROJECT_SOURCE_DIR}/bindings/coo.cpp
)

    target_include_directories(pygko PRIVATE ${Ginkgo_BINARY_DIR}/include)
    target_link_libraries(pygko PRIVATE Ginkgo::ginkgo)
    set_property(TARGET pygko PROPERTY CXX_STANDARD 14)

add_subdirectory(bindings/test)
