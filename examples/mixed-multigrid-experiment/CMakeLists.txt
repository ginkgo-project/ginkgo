cmake_minimum_required(VERSION 3.9)
project(mixed-multigrid-experiment)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

add_executable(mixed-multigrid-experiment mixed-multigrid-experiment.cpp)
target_link_libraries(mixed-multigrid-experiment Ginkgo::ginkgo ${CUDA_RUNTIME_LIBS})
message("CUDA_INCLUDE_DIRS ${CUDA_INCLUDE_DIRS}")
target_include_directories(mixed-multigrid-experiment PRIVATE "${CUDA_INCLUDE_DIRS}")

# Copy the data files to the execution directory
configure_file(data/A.mtx data/A.mtx COPYONLY)
configure_file(collect_case.sh collect_case.sh COPYONLY)
configure_file(run_selected.sh run_selected.sh COPYONLY)
