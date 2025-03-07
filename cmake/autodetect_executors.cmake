set(GINKGO_HAS_OMP OFF)
set(GINKGO_HAS_MPI OFF)
set(GINKGO_HAS_CUDA OFF)
set(GINKGO_HAS_SYCL OFF)
set(GINKGO_HAS_HIP OFF)
set(GINKGO_HAS_CSL OFF)

include(CheckLanguage)

if(NOT DEFINED GINKGO_BUILD_OMP)
    find_package(OpenMP 3.0)
    if(OpenMP_CXX_FOUND)
        message(STATUS "Enabling OpenMP executor")
        set(GINKGO_HAS_OMP ON)
    endif()
endif()

if(NOT DEFINED GINKGO_BUILD_MPI)
    find_package(MPI 3.1 COMPONENTS CXX)
    if(MPI_FOUND)
        message(STATUS "Enabling MPI support")
        set(GINKGO_HAS_MPI ON)
    endif()
endif()

if(NOT DEFINED GINKGO_BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        message(STATUS "Enabling CUDA executor")
        set(GINKGO_HAS_CUDA ON)
    endif()
endif()

if(NOT DEFINED GINKGO_BUILD_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        message(STATUS "Enabling HIP executor")
        set(GINKGO_HAS_HIP ON)
    else()
        include(cmake/hip_helpers.cmake)
        ginkgo_check_hip_detection_issue()
    endif()
endif()

if(NOT DEFINED GINKGO_BUILD_DPCPP AND NOT DEFINED GINKGO_BUILD_SYCL)
    try_compile(
        GKO_CAN_COMPILE_DPCPP
        ${PROJECT_BINARY_DIR}/dpcpp
        SOURCES ${PROJECT_SOURCE_DIR}/dpcpp/test_dpcpp.dp.cpp
        # try_compile will pass the project CMAKE_CXX_FLAGS so passing -DCMAKE_CXX_FLAGS does not affect it.
        # They append COMPILE_DEFINITIONS into CMAKE_CXX_FLAGS.
        # Note. it is different from try_compile COMPILE_DEFINITIONS affect
        CMAKE_FLAGS -DCOMPILE_DEFINITIONS=-fsycl
        CXX_STANDARD 17
    )
    if(GKO_CAN_COMPILE_DPCPP)
        message(STATUS "Enabling DPCPP executor")
        set(GINKGO_HAS_SYCL ON)
    endif()
endif()

if(NOT DEFINED GINKGO_BUILD_CSL)
    try_compile(
        GKO_CAN_COMPILE_CSL
        ${PROJECT_BINARY_DIR}/csl
        SOURCES ${PROJECT_SOURCE_DIR}/csl/test_csl.cpp
    )
    if(GKO_CAN_COMPILE_CSL)
        message(STATUS "Enabling Csl executor for Graphcore IPUs")
        set(GINKGO_HAS_CSL ON)
    endif()
endif()
