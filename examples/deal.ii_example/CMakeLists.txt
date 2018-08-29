# This is just an example of the CMakeLists.txt file that can be used after the
# correct version of deal.ii has been installed.
cmake_minimum_required(VERSION 3.8)
project(DEAL_II_EXAMPLE LANGUAGES CXX)

find_package(MPI REQUIRED)

set(deal.II_DIR "/path/to/deal.ii/installation")
find_package(deal.II 9.0.0 REQUIRED
  HINTS ${deal.II_DIR} ${DEAL_II_DIR})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Set the appropriate source 
set(SOURCE "./")
set(CMAKE_CXX_FLAGS "-g -Wall")

set(GINKGO_LIB_DIR "/path/to/ginkgo/lib/dir")
set(GINKGO_INC_DIR "/path/to/ginkgo/inc/dir")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_definitions(-DOMPI_SKIP_MPICXX)

add_executable(${PROJECT_NAME} ${SOURCE}/step-9.cc)

target_link_libraries(${PROJECT_NAME}
    ${MPI_C_LIBRARIES} ${GINKGO_LIB_DIR}/libginkgo.so ${GINKGO_LIB_DIR}/libginkgo_omp.so ${GINKGO_LIB_DIR}/libginkgo_cuda.so ${GINKGO_LIB_DIR}/libginkgo_reference.so )

target_include_directories(${PROJECT_NAME}
     PRIVATE ${MPI_C_INCLUDE_PATH} ${GINKGO_INC_DIR} ${GINKGO_LIB_DIR} )

DEAL_II_SETUP_TARGET(${PROJECT_NAME})

