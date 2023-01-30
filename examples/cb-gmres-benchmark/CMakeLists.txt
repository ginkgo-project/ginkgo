cmake_minimum_required(VERSION 3.9)
project(cb-gmres-benchmark)

# We only need to find Ginkgo if we build this example stand-alone
if (NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.5.0 REQUIRED)
endif()

# Added for additional CB-GMRES compression
find_package(LibPressio REQUIRED)
find_package(LibPressioTools REQUIRED)
find_package(std_compat REQUIRED)
#find_package(frsz REQUIRED)
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(cb-gmres-benchmark cb-gmres-benchmark.cpp)
target_link_libraries(cb-gmres-benchmark Ginkgo::ginkgo
      # Added for additional compression in CB-GMRES
      nlohmann_json::nlohmann_json
      #frsz::frsz
      LibPressio::libpressio
      LibPressioTools::libpressio_meta
      OpenMP::OpenMP_CXX
    )

# Copy the data files to the execution directory
configure_file("${Ginkgo_SOURCE_DIR}/matrices/test/ani1.mtx" data/A.mtx COPYONLY)
