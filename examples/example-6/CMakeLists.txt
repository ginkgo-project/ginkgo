# This is just an example of the CMakeLists.txt file that can be used after the
# correct version of deal.ii has been installed.
cmake_minimum_required(VERSION 3.8)
project(DEAL_II_EXAMPLE LANGUAGES CXX)

find_package(MPI REQUIRED)

set(deal.II_DIR "/path/to/deal.ii/installation")
find_package(deal.II 9.0.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


add_executable(${PROJECT_NAME} "")
target_sources(${PROJECT_NAME} PRIVATE example-6.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE -g -Wall)
target_compile_definitions(${PROJECT_NAME} PRIVATE OMPI_SKIP_MPICXX)

target_link_libraries(${PROJECT_NAME}
    ${MPI_C_LIBRARIES} Ginkgo::ginkgo)

target_include_directories(${PROJECT_NAME}
     PRIVATE ${MPI_C_INCLUDE_PATH} ${GINKGO_INC_DIR} ${GINKGO_LIB_DIR} )

DEAL_II_SETUP_TARGET(${PROJECT_NAME})

