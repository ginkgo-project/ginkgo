set(gko_test_resource_args "RESOURCE_LOCAL_CORES;RESOURCE_TYPE")
set(gko_test_single_args "MPI_SIZE;EXECUTABLE_NAME;${gko_test_resource_args}")
set(gko_test_multi_args "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_INCLUDES")
set(gko_test_option_args "NO_RESOURCES")

## Replaces / by _ to create valid target names from relative paths
function(ginkgo_build_test_name test_name target_name)
    file(RELATIVE_PATH REL_BINARY_DIR
         ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    set(${target_name} ${TEST_TARGET_NAME} PARENT_SCOPE)
endfunction()

## Set up shared target properties and handle ADDITIONAL_LIBRARIES/ADDITIONAL_INCLUDES
## `MPI_SIZE size` causes the tests to be run with `size` MPI processes.
function(ginkgo_set_test_target_properties test_target_name test_library_suffix)
    cmake_parse_arguments(PARSE_ARGV 1 set_properties "" "${gko_test_single_args}" "${gko_test_multi_args}")
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_TEST_NONDEFAULT_STREAM)
        target_compile_definitions(${test_target_name} PRIVATE GKO_TEST_NONDEFAULT_STREAM)
    endif()
    if (GINKGO_COMPILING_DPCPP_TEST AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    if(GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${test_target_name} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    if(set_properties_MPI_SIZE)
        target_link_libraries(${test_target_name} PRIVATE ginkgo_gtest_main_mpi${test_library_suffix})
    else()
        target_link_libraries(${test_target_name} PRIVATE ginkgo_gtest_main${test_library_suffix})
    endif()
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${GINKGO_COMPILER_FLAGS}>)
    target_include_directories(${test_target_name} PRIVATE ${Ginkgo_BINARY_DIR} ${set_properties_ADDITIONAL_INCLUDES})
    target_link_libraries(${test_target_name} PRIVATE ginkgo GTest::GTest ${set_properties_ADDITIONAL_LIBRARIES})
endfunction()

function(ginkgo_add_resource_requirement test_name)
    cmake_parse_arguments(PARSE_ARGV 1 add_rr "${gko_test_option_args}" "${gko_test_single_args}" "")
    if(add_rr_NO_RESOURCES OR (NOT add_rr_RESOURCE_TYPE))
        return()
    endif ()

    if(add_rr_RESOURCE_TYPE STREQUAL "cpu")
        if(NOT add_rr_RESOURCE_LOCAL_CORES)
            set(add_rr_RESOURCE_LOCAL_CORES ${GINKGO_CI_TEST_OMP_PARALLELISM})
        endif()
        if(NOT add_rr_RESOURCE_LOCAL_CORES MATCHES "^[0-9]+")
            message(FATAL_ERROR "Resource specification is invalid: RESOURCE_LOCAL_CORES=${add_rr_RESOURCE_LOCAL_CORES}")
        endif()

        set(single_resource "cpu:${add_rr_RESOURCE_LOCAL_CORES}")
    elseif(add_rr_RESOURCE_TYPE MATCHES "^(cudagpu|hipgpu|sycl)$")
        set(single_resource "${add_rr_RESOURCE_TYPE}:1")
    else()
        message(FATAL_ERROR "Unrecognized resource type ${add_rr_RESOURCE_TYPE}, allowed are: cpu, cudagpu, hipgpu, sycl.")
    endif()

    if(NOT add_rr_MPI_SIZE)
        set(add_rr_MPI_SIZE 1)
    endif()
    set_property(TEST ${test_name}
                 PROPERTY
                 RESOURCE_GROUPS "${add_rr_MPI_SIZE},${single_resource}")
endfunction()


## Adds a test to the list executed by ctest and sets its output binary name
## Possible additional arguments:
## - `MPI_SIZE size` causes the tests to be run with `size` MPI processes.
## - `RESOURCE_LOCAL_CORES` the number of threads used by a test, default is
##    $GINKGO_CI_TEST_OMP_PARALLELISM
## - `RESOURCE_TYPE` the resource type, can be cpu, cudagpu, hipgpu, sycl
## - `DISABLE_EXECUTORS exec1 exec2` disables the test for certain backends (if built for multiple)
## - `ADDITIONAL_LIBRARIES lib1 lib2` adds additional target link dependencies
## - `ADDITIONAL_INCLUDES path1 path2` adds additional target include paths
function(ginkgo_add_test test_name test_target_name)
    cmake_parse_arguments(PARSE_ARGV 2 add_test "" "${gko_test_single_args}" "${gko_test_multi_args}")
    file(RELATIVE_PATH REL_BINARY_DIR ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    set(test_binary_name ${test_name})
    if (add_test_EXECUTABLE_NAME)
        set(test_binary_name ${add_test_EXECUTABLE_NAME})
    endif()
    set_target_properties(${test_target_name} PROPERTIES OUTPUT_NAME ${test_binary_name})
    if (add_test_MPI_SIZE)
        add_test(NAME ${REL_BINARY_DIR}/${test_binary_name}
                 COMMAND
                     ${MPIEXEC_EXECUTABLE}
                     ${MPIEXEC_NUMPROC_FLAG}
                     ${add_test_MPI_SIZE}
                     "$<TARGET_FILE:${test_target_name}>"
                 WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    else()
        add_test(NAME ${REL_BINARY_DIR}/${test_binary_name}
                 COMMAND ${test_target_name}
                 WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    endif()

    ginkgo_add_resource_requirement(${REL_BINARY_DIR}/${test_binary_name} ${ARGN})

    set(test_preload)
    if (GINKGO_TEST_NONDEFAULT_STREAM AND GINKGO_BUILD_CUDA)
        set(test_preload $<TARGET_FILE:identify_stream_usage_cuda>:${test_preload})
    endif()
    if (GINKGO_TEST_NONDEFAULT_STREAM AND GINKGO_BUILD_HIP AND GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_AMD_REGEX}")
        set(test_preload $<TARGET_FILE:identify_stream_usage_hip>:${test_preload})
    endif()
    if(test_preload)
        set_tests_properties(${REL_BINARY_DIR}/${test_binary_name} PROPERTIES ENVIRONMENT LD_PRELOAD=${test_preload})
    endif()
endfunction()

## Normal test
function(ginkgo_create_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_link_libraries(${test_target_name})
    ginkgo_set_test_target_properties(${test_target_name} "_cpu" ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN})
endfunction(ginkgo_create_test)

## Test compiled with dpcpp
function(ginkgo_create_dpcpp_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.dp.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_17)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_DPCPP_FLAGS})
    gko_add_sycl_to_target(TARGET ${test_target_name} SOURCES ${test_name}.dp.cpp)
    target_link_options(${test_target_name} PRIVATE -fsycl-device-code-split=per_kernel)
    ginkgo_set_test_target_properties(${test_target_name} "_dpcpp" ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN} RESOURCE_TYPE sycl)
    # Note: MKL_ENV is empty on linux. Maybe need to apply MKL_ENV to all test.
    if (MKL_ENV)
        set_tests_properties(${test_target_name} PROPERTIES ENVIRONMENT "${MKL_ENV}")
    endif()
endfunction(ginkgo_create_dpcpp_test)

## Test compiled with CUDA
function(ginkgo_create_cuda_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    ginkgo_create_cuda_test_internal(${test_name} ${test_name}.cu ${test_target_name} ${ARGN})
endfunction(ginkgo_create_cuda_test)

## Internal function allowing separate test name, filename and target name
function(ginkgo_create_cuda_test_internal test_name filename test_target_name)
    add_executable(${test_target_name} ${filename})
    target_compile_definitions(${test_target_name} PRIVATE GKO_COMPILING_CUDA)
    target_compile_options(${test_target_name}
        PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_COMPILER_FLAGS}>)
    if(MSVC)
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    endif()
    # we handle CUDA architecture flags for now, disable CMake handling
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_target_properties(${test_target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
    ginkgo_set_test_target_properties(${test_target_name} "_cuda" ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN} RESOURCE_TYPE cudagpu)
endfunction(ginkgo_create_cuda_test_internal)

## Test compiled with HIP
function(ginkgo_create_hip_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    ginkgo_create_hip_test_internal(${test_name} ${test_name}.hip.cpp ${test_target_name} "" ${ARGN})
endfunction(ginkgo_create_hip_test)

## Internal function allowing separate filename, test name and test target name.
function(ginkgo_create_hip_test_internal test_name filename test_target_name additional_flags)
    set_source_files_properties(${filename} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    set(GINKGO_TEST_HIP_DEFINES -DGKO_COMPILING_HIP ${additional_flags})
    if (GINKGO_FAST_TESTS)
        list(APPEND GINKGO_TEST_HIP_DEFINES -DGINKGO_FAST_TESTS)
    endif()
    if (GINKGO_TEST_NONDEFAULT_STREAM)
        list(APPEND GINKGO_TEST_HIP_DEFINES -DGKO_TEST_NONDEFAULT_STREAM)
    endif()

    # NOTE: With how HIP works, passing the flags `HIPCC_OPTIONS` etc. here
    # creates a redefinition of all flags. This creates some issues with `nvcc`,
    # but `clang` seems fine with the redefinitions.
    if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
        hip_add_executable(${test_target_name} ${filename}
            # If `FindHIP.cmake`, namely `HIP_PARSE_HIPCC_OPTIONS` macro and
            # call gets fixed, uncomment this.
            HIPCC_OPTIONS ${GINKGO_TEST_HIP_DEFINES} # ${GINKGO_HIPCC_OPTIONS}
            # NVCC_OPTIONS  ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_NVCC_OPTIONS}
            # CLANG_OPTIONS ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_CLANG_OPTIONS}
            --expt-relaxed-constexpr --expt-extended-lambda
            )
    else() # hcc/clang
        hip_add_executable(${test_target_name} ${filename}
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} ${GINKGO_TEST_HIP_DEFINES}
            NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
            CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
            )
    endif()

    # Let's use a normal compiler for linking
    set_target_properties(${test_target_name} PROPERTIES LINKER_LANGUAGE CXX)

    target_include_directories(${test_target_name}
        PRIVATE
        # Only `math` requires it so far, but it's much easier
        # to put these this way.
        ${GINKGO_HIP_THRUST_PATH}
        # Only `exception_helpers` requires these so far, but it's much easier
        # to put these this way.
        ${HIPBLAS_INCLUDE_DIRS}
        ${HIPFFT_INCLUDE_DIRS}
        ${hiprand_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        )
    ginkgo_set_test_target_properties(${test_target_name} "_hip" ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN} RESOURCE_TYPE hipgpu)
endfunction(ginkgo_create_hip_test_internal)


## Test compiled with OpenMP
function(ginkgo_create_omp_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    ginkgo_create_omp_test_internal(${test_name} ${test_name}.cpp ${test_target_name} "" ${ARGN})
endfunction()

function(ginkgo_create_omp_test_internal test_name filename test_target_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_definitions(${test_target_name} PRIVATE GKO_COMPILING_OMP)
    target_link_libraries(${test_target_name} PRIVATE OpenMP::OpenMP_CXX)
    ginkgo_set_test_target_properties(${test_target_name} "_omp" ${ARGN})
    ginkgo_add_test(${test_name} ${test_target_name} ${ARGN} RESOURCE_TYPE cpu)
endfunction()

## Common test compiled with the host compiler, one target for each enabled backend
function(ginkgo_create_common_test test_name)
    if(GINKGO_BUILD_OMP)
        ginkgo_create_common_test_internal(${test_name} OmpExecutor omp ${ARGN})
    endif()
    if(GINKGO_BUILD_HIP)
        ginkgo_create_common_test_internal(${test_name} HipExecutor hip ${ARGN})
    endif()
    if(GINKGO_BUILD_CUDA)
        ginkgo_create_common_test_internal(${test_name} CudaExecutor cuda ${ARGN})
    endif()
    if(GINKGO_BUILD_SYCL)
        ginkgo_create_common_test_internal(${test_name} DpcppExecutor dpcpp ${ARGN})
    endif()
endfunction(ginkgo_create_common_test)

function(ginkgo_create_common_test_internal test_name exec_type exec)
    cmake_parse_arguments(PARSE_ARGV 3 common_test "" "${gko_test_single_args}" "${gko_test_multi_args}")
    if(exec IN_LIST common_test_DISABLE_EXECUTORS)
        return()
    endif()
    if (exec STREQUAL reference)
        set(test_resource_type "")
    elseif (exec STREQUAL omp)
        set(test_resource_type cpu)
    elseif (exec STREQUAL cuda)
        set(test_resource_type cudagpu)
    elseif (exec STREQUAL hip)
        set(test_resource_type hipgpu)
    else ()
        set(test_resource_type sycl)
    endif ()
    ginkgo_build_test_name(${test_name} test_target_name)
    string(TOUPPER ${exec} exec_upper)

    # set up actual test
    set(test_target_name ${test_target_name}_${exec})
    add_executable(${test_target_name} ${test_name}.cpp)

    # also need to add runtime libraries for other backends
    if (exec STREQUAL omp)
        target_link_libraries(${test_target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    target_compile_definitions(${test_target_name} PRIVATE EXEC_TYPE=${exec_type} EXEC_NAMESPACE=${exec} GKO_COMPILING_${exec_upper})
    target_link_libraries(${test_target_name} PRIVATE ${common_test_ADDITIONAL_LIBRARIES})
    # use float for DPC++ if necessary
    if((exec STREQUAL "dpcpp") AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_COMMON_SINGLE_MODE=1)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    ginkgo_set_test_target_properties(${test_target_name} "_${exec}" ${ARGN})
    ginkgo_add_test(${test_name}_${exec} ${test_target_name} ${ARGN} RESOURCE_TYPE ${test_resource_type})
endfunction(ginkgo_create_common_test_internal)

## Common test compiled with the device compiler, one target for each enabled backend
function(ginkgo_create_common_device_test test_name)
    cmake_parse_arguments(PARSE_ARGV 1 common_device_test "" "${gko_test_single_args}" "${gko_test_multi_args}")
    ginkgo_build_test_name(${test_name} test_target_name)
    if(GINKGO_BUILD_SYCL)
        ginkgo_create_common_test_internal(${test_name} DpcppExecutor dpcpp ${ARGN})
        target_compile_features(${test_target_name}_dpcpp PRIVATE cxx_std_17)
        target_compile_options(${test_target_name}_dpcpp PRIVATE ${GINKGO_DPCPP_FLAGS})
        # We need to use a new file to avoid sycl setting in other backends because add_sycl_to_target will change the source property.
        configure_file(${test_name}.cpp ${test_name}.dp.cpp COPYONLY)
        gko_add_sycl_to_target(TARGET ${test_target_name}_dpcpp SOURCES ${test_name}.dp.cpp)
        target_link_options(${test_target_name}_dpcpp PRIVATE -fsycl-device-lib=all -fsycl-device-code-split=per_kernel)
    endif()
    if(GINKGO_BUILD_OMP)
        ginkgo_create_common_test_internal(${test_name} OmpExecutor omp ${ARGN})
        target_link_libraries(${test_target_name}_omp PUBLIC OpenMP::OpenMP_CXX)
    endif()
    if(GINKGO_BUILD_CUDA)
        # need to make a separate file for this, since we can't set conflicting properties on the same file
        configure_file(${test_name}.cpp ${test_name}.cu COPYONLY)
        ginkgo_create_cuda_test_internal(${test_name}_cuda ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.cu ${test_target_name}_cuda ${ARGN})
        target_compile_definitions(${test_target_name}_cuda PRIVATE EXEC_TYPE=CudaExecutor EXEC_NAMESPACE=cuda)
    endif()
    if(GINKGO_BUILD_HIP)
        # need to make a separate file for this, since we can't set conflicting properties on the same file
        configure_file(${test_name}.cpp ${test_name}.hip.cpp COPYONLY)
        ginkgo_create_hip_test_internal(${test_name}_hip ${CMAKE_CURRENT_BINARY_DIR}/${test_name}.hip.cpp ${test_target_name}_hip "-std=c++14;-DEXEC_TYPE=HipExecutor;-DEXEC_NAMESPACE=hip" ${ARGN})
    endif()
endfunction(ginkgo_create_common_device_test)

## Common test compiled with the host compiler for all enabled backends and Reference
function(ginkgo_create_common_and_reference_test test_name)
    ginkgo_create_common_test(${test_name} ${ARGN})
    ginkgo_create_common_test_internal(${test_name} ReferenceExecutor reference REFERENCE ${ARGN})
endfunction(ginkgo_create_common_and_reference_test)
