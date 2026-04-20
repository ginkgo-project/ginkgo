if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

find_package(cudss 0.7.1 CONFIG QUIET)
if(NOT cudss_FOUND)
    message(
        STATUS
        "cuDSS not found — gko::ext::cuda::solver::CuDss will not be built"
    )
    return()
endif()

message(STATUS "Building ginkgo_cudss extension")

add_library(ginkgo_cudss solver/cudss.cpp)

target_link_libraries(ginkgo_cudss PRIVATE cudss CUDA::cudart)

# Use the standard Ginkgo include path setup (public headers)
ginkgo_default_includes(ginkgo_cudss)

ginkgo_install_library(ginkgo_cudss)

# Absorb into the umbrella ginkgo target (same pattern as ginkgo_cuda etc.)
# extensions/ is add_subdirectory'd after core/, so the ginkgo target exists.
target_link_libraries(ginkgo PUBLIC ginkgo_cudss)
