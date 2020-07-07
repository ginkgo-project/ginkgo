if (GINKGO_CHECK_CIRCULAR_DEPS)
    add_library(ginkgo_public_api INTERFACE) # dummy target
    set_property(TARGET ginkgo_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}")
    set_property(TARGET ginkgo_public_api APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_BINARY_DIR}")
    ginkgo_check_headers(ginkgo_public_api)
endif()
