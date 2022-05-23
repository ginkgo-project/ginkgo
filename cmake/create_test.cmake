function(ginkgo_build_test_name test_name target_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    set(${target_name} ${TEST_TARGET_NAME} PARENT_SCOPE)
endfunction(ginkgo_build_test_name)

function(ginkgo_create_gtest_mpi_main)
    add_library(gtest_mpi_main "")
    target_sources(gtest_mpi_main
        PRIVATE
        ${PROJECT_SOURCE_DIR}/core/test/mpi/gtest/mpi_listener.cpp)
    find_package(MPI REQUIRED)
    target_link_libraries(gtest_mpi_main PRIVATE GTest::GTest MPI::MPI_CXX)
endfunction(ginkgo_create_gtest_mpi_main)

function(ginkgo_set_test_target_default_properties test_name test_target_name)
    set_target_properties(${test_target_name} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_COMPILING_DPCPP_TEST AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${test_target_name} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_include_directories(${test_target_name} PRIVATE ${Ginkgo_BINARY_DIR})
    target_link_libraries(${test_target_name} PRIVATE ginkgo)
endfunction(ginkgo_set_test_target_default_properties)

function(ginkgo_internal_add_test test_name test_target_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    ginkgo_set_test_target_default_properties(${test_name} ${test_target_name})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${test_target_name}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
    target_link_libraries(${test_target_name} PRIVATE GTest::Main GTest::GTest)
endfunction(ginkgo_internal_add_test)

function(ginkgo_internal_add_mpi_test test_name test_target_name num_mpi_procs)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    ginkgo_set_test_target_default_properties(${test_name} ${test_target_name})
    if(NOT TARGET gtest_mpi_main)
        ginkgo_create_gtest_mpi_main()
    endif()
    target_link_libraries(${test_target_name} PRIVATE gtest_mpi_main GTest::GTest MPI::MPI_CXX)
    set(test_param ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_procs} ${OPENMPI_RUN_AS_ROOT_FLAG}
                   ${CMAKE_BINARY_DIR}/${REL_BINARY_DIR}/${test_name})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${MPIEXEC_EXECUTABLE} ${test_param})
endfunction(ginkgo_internal_add_mpi_test)

function(ginkgo_create_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_test(${test_name} ${test_target_name})
endfunction(ginkgo_create_test)

function(ginkgo_create_dpcpp_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.dp.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_17)
    target_compile_options(${test_target_name} PRIVATE "${GINKGO_DPCPP_FLAGS}")
    target_compile_options(${test_target_name} PRIVATE "${GINKGO_COMPILER_FLAGS}")
    target_link_options(${test_target_name} PRIVATE -fsycl-device-code-split=per_kernel)
    ginkgo_internal_add_test(${test_name} ${test_target_name})
    # Note: MKL_ENV is empty on linux. Maybe need to apply MKL_ENV to all test.
    if (MKL_ENV)
       set_tests_properties(${test_target_name} PROPERTIES ENVIRONMENT "${MKL_ENV}")
    endif()
endfunction(ginkgo_create_dpcpp_test)

function(ginkgo_create_thread_test test_name)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_link_libraries(${test_target_name} PRIVATE Threads::Threads ${ARGN})
    ginkgo_internal_add_test(${test_name} ${test_target_name})
endfunction(ginkgo_create_thread_test)

function(ginkgo_create_mpi_test test_name num_mpi_procs)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_mpi_test(${test_name} ${test_target_name} ${num_mpi_procs})
endfunction(ginkgo_create_mpi_test)

function(ginkgo_create_test_cpp_cuda_header test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories(${test_target_name} PRIVATE "${CUDA_INCLUDE_DIRS}")
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_test(${test_name} ${test_target_name})
endfunction(ginkgo_create_test_cpp_cuda_header)

function(ginkgo_create_cuda_test test_name)
    ginkgo_build_test_name(${test_name} test_target_name)
    add_executable(${test_target_name} ${test_name}.cu)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name}
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
    if(MSVC)
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>)
    elseif(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
        target_compile_options(${test_target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda --expt-relaxed-constexpr>)
    endif()
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    # we handle CUDA architecture flags for now, disable CMake handling
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
        set_target_properties(${test_target_name} PROPERTIES CUDA_ARCHITECTURES OFF)
    endif()
    ginkgo_internal_add_test(${test_name} ${test_target_name})
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_hip_test test_name)
ginkgo_build_test_name(${test_name} test_target_name)
    set_source_files_properties(${test_name}.hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    set(GINKGO_TEST_HIP_DEFINES)
    if (GINKGO_FAST_TESTS)
        set(GINKGO_TEST_HIP_DEFINES -DGINKGO_FAST_TESTS)
    endif()

    # NOTE: With how HIP works, passing the flags `HIPCC_OPTIONS` etc. here
    # creates a redefinition of all flags. This creates some issues with `nvcc`,
    # but `clang` seems fine with the redefinitions.
    if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
        hip_add_executable(${test_target_name} ${test_name}.hip.cpp
            # If `FindHIP.cmake`, namely `HIP_PARSE_HIPCC_OPTIONS` macro and
            # call gets fixed, uncomment this.
            HIPCC_OPTIONS ${GINKGO_TEST_HIP_DEFINES} # ${GINKGO_HIPCC_OPTIONS}
            # NVCC_OPTIONS  ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_NVCC_OPTIONS}
            # CLANG_OPTIONS ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_CLANG_OPTIONS}
            --expt-relaxed-constexpr --expt-extended-lambda
            )
    else() # hcc/clang
        hip_add_executable(${test_target_name} ${test_name}.hip.cpp
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
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_test(${test_name} ${test_target_name})
endfunction(ginkgo_create_hip_test)

function(ginkgo_internal_create_common_test_template test_name)
    cmake_parse_arguments(PARSE_ARGV 1 common_test "" "TEST_TYPE" "DISABLE_EXECUTORS;ADDITIONAL_LIBRARIES;ADDITIONAL_TEST_PARAMETERS")
    string(TOLOWER ${common_test_TEST_TYPE} test_type)
    set(executors)
    if(GINKGO_BUILD_OMP)
        list(APPEND executors omp)
    endif()
    if(GINKGO_BUILD_HIP)
        list(APPEND executors hip)
    endif()
    if(GINKGO_BUILD_CUDA)
        list(APPEND executors cuda)
    endif()
    if(GINKGO_BUILD_DPCPP)
        list(APPEND executors dpcpp)
    endif()
    foreach(disabled_exec ${common_test_DISABLE_EXECUTORS})
        list(REMOVE_ITEM executors ${disabled_exec})
    endforeach()
    foreach(exec ${executors})
        ginkgo_build_test_name(${test_name} test_target_name)
        # build executor typename out of shorthand
        string(SUBSTRING ${exec} 0 1 exec_initial)
        string(SUBSTRING ${exec} 1 -1 exec_tail)
        string(TOUPPER ${exec_initial} exec_initial)
        string(TOUPPER ${exec} exec_upper)
        set(exec_type ${exec_initial}${exec_tail}Executor)
        # set up actual test
        set(test_target_name ${test_target_name}_${exec})
        add_executable(${test_target_name} ${test_name}.cpp)
        target_compile_features(${test_target_name} PUBLIC cxx_std_14)
        target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
        target_compile_definitions(${test_target_name} PRIVATE EXEC_TYPE=${exec_type} EXEC_NAMESPACE=${exec} GKO_COMPILING_${exec_upper})
        target_link_libraries(${test_target_name} PRIVATE ${common_test_ADDITIONAL_LIBRARIES})
        # use float for DPC++ if necessary
        if((exec STREQUAL "dpcpp") AND GINKGO_DPCPP_SINGLE_MODE)
            target_compile_definitions(${test_target_name} PRIVATE GINKGO_COMMON_SINGLE_MODE=1)
            target_compile_definitions(${test_target_name} PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
        endif()
        if(${test_type} STREQUAL default)
            ginkgo_internal_add_test(${test_name}_${exec} ${test_target_name})
        elseif(${test_type} STREQUAL mpi)
            ginkgo_internal_add_mpi_test(${test_name}_${exec} ${test_target_name} ${common_test_ADDITIONAL_TEST_PARAMETERS})
        else()
            message(FATAL_ERROR "Encountered unrecognized test type ${test_type} during common test creation.")
        endif()
    endforeach()
endfunction(ginkgo_internal_create_common_test_template)

function(ginkgo_create_common_test test_name)
    ginkgo_internal_create_common_test_template(${test_name} TEST_TYPE default ${ARGN})
endfunction(ginkgo_create_common_test)

function(ginkgo_create_common_mpi_test test_name num_mpi_procs)
    ginkgo_internal_create_common_test_template(${test_name} TEST_TYPE mpi ADDITIONAL_TEST_PARAMETERS ${num_mpi_procs} ${ARGN})
endfunction(ginkgo_create_common_mpi_test)

function(ginkgo_create_common_and_reference_test test_name)
    ginkgo_create_common_test(${test_name})
    ginkgo_build_test_name(${test_name} test_target_name)
    set(test_target_name ${test_target_name}_reference)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_compile_definitions(${test_target_name} PRIVATE EXEC_TYPE=ReferenceExecutor EXEC_NAMESPACE=reference GKO_COMPILING_REFERENCE)
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_test(${test_name}_reference ${test_target_name})
endfunction()


function(ginkgo_create_common_and_reference_mpi_test test_name num_mpi_procs)
    ginkgo_create_common_mpi_test(${test_name} ${num_mpi_procs})
    ginkgo_build_test_name(${test_name} test_target_name)
    set(test_target_name ${test_target_name}_reference)
    add_executable(${test_target_name} ${test_name}.cpp)
    target_compile_features(${test_target_name} PUBLIC cxx_std_14)
    target_compile_options(${test_target_name} PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_compile_definitions(${test_target_name} PRIVATE EXEC_TYPE=ReferenceExecutor EXEC_NAMESPACE=reference GKO_COMPILING_REFERENCE)
    target_link_libraries(${test_target_name} PRIVATE ${ARGN})
    ginkgo_internal_add_mpi_test(${test_name}_reference ${test_target_name} ${num_mpi_procs})
endfunction()
