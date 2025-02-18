# Kokkos doesn't handle any compiler launcher well, so it's disabled for these tests
set(_CMAKE_CXX_COMPILER_LAUNCHER ${CMAKE_CXX_COMPILER_LAUNCHER})
unset(CMAKE_CXX_COMPILER_LAUNCHER)

function(_ginkgo_check_build_config _build)
    string(TOUPPER ${_build} _build)
    if(NOT GINKGO_BUILD_${_build})
        message(
            FATAL_ERROR
            "Building test with ${_build} enabled for Kokkos, but not for Ginkgo."
        )
    endif()
    unset(_build)
endfunction()

if(Kokkos_ENABLE_CUDA)
    _ginkgo_check_build_config(cuda)
    set(resource_type "cudagpu")
    set(definitions "GKO_COMPILING_CUDA")
elseif(Kokkos_ENABLE_HIP)
    _ginkgo_check_build_config(hip)
    set(resource_type "hipgpu")
    set(definitions "GKO_COMPILING_HIP")
elseif(Kokkos_ENABLE_SYCL)
    _ginkgo_check_build_config(sycl)
    set(resource_type "sycl")
    set(definitions "GKO_COMPILING_DPCPP")
else()
    set(resource_type "cpu")
    if(Kokkos_ENABLE_OPENMP)
        _ginkgo_check_build_config(omp)
        set(definitions "GKO_COMPILING_OMP")
    endif()
endif()

function(create_gtest_main_kokkos)
    add_library(
        ginkgo_gtest_main_kokkos
        STATIC
        kokkos_main.cpp
        ${PROJECT_SOURCE_DIR}/core/test/gtest/resources.cpp
    )
    target_link_libraries(
        ginkgo_gtest_main_kokkos
        PUBLIC Ginkgo::ginkgo GTest::GTest Kokkos::kokkos
    )
    target_compile_definitions(ginkgo_gtest_main_kokkos PRIVATE ${definitions})
    ginkgo_compile_features(ginkgo_gtest_main_kokkos)
endfunction()
create_gtest_main_kokkos()

function(ginkgo_create_test_kokkos test_name)
    ginkgo_create_test(${test_name} NO_GTEST_MAIN RESOURCE_TYPE ${resource_type} ADDITIONAL_LIBRARIES Kokkos::kokkos ginkgo_gtest_main_kokkos ${ARGN})
endfunction()

ginkgo_create_test_kokkos(types)
ginkgo_create_test_kokkos(spaces)

# restore the compiler launcher variable
set(CMAKE_CXX_COMPILER_LAUNCHER ${_CMAKE_CXX_COMPILER_LAUNCHER} PARENT_SCOPE)
