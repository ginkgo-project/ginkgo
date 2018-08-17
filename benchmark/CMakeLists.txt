if (NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING
        "Ginko is not being built in \"Release\" mode, benchmark performance "
        "will be affected")
endif()
add_subdirectory(spmv_comparison_cuda)
add_subdirectory(solver)
