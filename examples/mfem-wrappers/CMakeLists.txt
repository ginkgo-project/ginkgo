if(GINKGO_BUILD_MFEM_WRAP_EXAMPLE)
# This is just an example of the CMakeLists.txt file that can be used after the
# correct version of MFEM has been installed.
cmake_minimum_required(VERSION 3.8)
project(mfem-wrappers LANGUAGES CXX)

find_package(MPI REQUIRED)

set(MFEM_DIR "/path/to/mfem/installation")
find_package(MFEM 4.0 REQUIRED
  HINTS ${MFEM_DIR})

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

add_executable(${PROJECT_NAME} test_mfem.cpp)
target_sources(${PROJECT_NAME} PRIVATE test_mfem.cpp mfem_wrapper.cpp)
target_compile_options(${PROJECT_NAME} PRIVATE -g -Wall -fopenmp)

target_link_libraries(${PROJECT_NAME}
         PRIVATE ${MPI_C_LIBRARIES} ${MFEM_LIBRARIES} Ginkgo::ginkgo -fopenmp)

target_include_directories(${PROJECT_NAME}
         PRIVATE ${MPI_C_INCLUDE_PATH} ${GINKGO_INC_DIR} ${GINKGO_LIB_DIR} 
                ${MFEM_INCLUDE_DIRS} ${MFEM_LIB_DIR})

endif()
