if(NOT GINKGO_BUILD_CUDA)
    return()
endif()

if(NOT cudss_FOUND)
    message(
        STATUS
        "cuDSS not found — gko::ext::cuda::solver::Cudss will not be built"
    )
    return()
endif()

message(STATUS "Building ginkgo_cudss extension")

add_library(ginkgo_cudss solver/cudss.cpp)
add_library(Ginkgo::ginkgo_cudss ALIAS ginkgo_cudss)

target_link_libraries(ginkgo_cudss PUBLIC ginkgo)
target_link_libraries(ginkgo_cudss PRIVATE cudss CUDA::cudart)

# Use the standard Ginkgo include path setup (public headers)
ginkgo_default_includes(ginkgo_cudss)

ginkgo_install_library(ginkgo_cudss)
