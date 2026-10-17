cmake_minimum_required(VERSION 3.16)
project(workspace-reuse)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

add_executable(workspace-reuse workspace-reuse.cpp)
target_link_libraries(workspace-reuse Ginkgo::ginkgo)

# The cudss option needs the cuDSS extension of Ginkgo
if(TARGET Ginkgo::ginkgo_cudss)
    target_link_libraries(workspace-reuse Ginkgo::ginkgo_cudss)
    target_compile_definitions(workspace-reuse PRIVATE GKO_EXAMPLE_HAVE_CUDSS)
endif()
