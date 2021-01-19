find_package(MPI REQUIRED)

add_library(ginkgo_mpi $<TARGET_OBJECTS:ginkgo_mpi_device> "")
target_sources(ginkgo_mpi
    PRIVATE
        base/exception.cpp
        base/executor.cpp
        base/version.cpp
        )

ginkgo_compile_features(ginkgo_mpi)
target_include_directories(ginkgo_mpi
  SYSTEM PRIVATE ${MPI_INCLUDE_PATH})
target_link_libraries(ginkgo_mpi PRIVATE "${MPI_C_LIBRARIES}" "${MPI_CXX_LIBRARIES}")
target_compile_options(ginkgo_mpi PRIVATE "${GINKGO_COMPILER_FLAGS}")

ginkgo_default_includes(ginkgo_mpi)
ginkgo_install_library(ginkgo_mpi mpi)

if (GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_mpi)
endif()

if(GINKGO_BUILD_TESTS)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
  add_subdirectory(test)
endif()

# Propagate some useful information
set(MPI_C_VERSION ${MPI_C_VERSION} PARENT_SCOPE)
set(MPI_C_LIBRARIES ${MPI_C_LIBRARIES} PARENT_SCOPE)
