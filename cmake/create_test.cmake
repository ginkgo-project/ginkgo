function(ginkgo_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_COMPILING_DPCPP_TEST AND GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions("${TEST_TARGET_NAME}" PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_test)

function(ginkgo_create_dpcpp_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.dp.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_17)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE "${GINKGO_DPCPP_FLAGS}")
    if (GINKGO_DPCPP_SINGLE_MODE)
        target_compile_definitions("${TEST_TARGET_NAME}" PRIVATE GINKGO_DPCPP_SINGLE_MODE=1)
    endif()
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_dpcpp_test)

function(ginkgo_create_mpi_test test_name num_mpi_procs)
  file(RELATIVE_PATH REL_BINARY_DIR
    ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
  string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
  add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
  target_include_directories("${TEST_TARGET_NAME}"
    PRIVATE
    "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
    ${MPI_INCLUDE_PATH}
    )
  set_target_properties(${TEST_TARGET_NAME} PROPERTIES
    OUTPUT_NAME ${test_name})
  if (GINKGO_CHECK_CIRCULAR_DEPS)
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
  endif()
  if("${GINKGO_MPI_EXEC_SUFFIX}" MATCHES ".openmpi" AND MPI_RUN_AS_ROOT)
      set(OPENMPI_RUN_AS_ROOT_FLAG "--allow-run-as-root")
  else()
      set(OPENMPI_RUN_AS_ROOT_FLAG "")
  endif()
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
  target_link_libraries(${TEST_TARGET_NAME} PRIVATE ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(test_param ${MPIEXEC_NUMPROC_FLAG} ${num_mpi_procs} ${OPENMPI_RUN_AS_ROOT_FLAG} ${CMAKE_BINARY_DIR}/${REL_BINARY_DIR}/${test_name})
  add_test(NAME ${REL_BINARY_DIR}/${test_name}
    COMMAND ${MPIEXEC_EXECUTABLE} ${test_param} )
endfunction(ginkgo_create_mpi_test)

function(ginkgo_create_thread_test test_name)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest
        Threads::Threads ${ARGN})
        add_test(NAME ${REL_BINARY_DIR}/${test_name}
            COMMAND ${TEST_TARGET_NAME}
            WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_thread_test)

function(ginkgo_create_test_cpp_cuda_header test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_compile_options("${TEST_TARGET_NAME}" PRIVATE ${GINKGO_COMPILER_FLAGS})
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        "${CUDA_INCLUDE_DIRS}"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_test_cpp_cuda_header)

function(ginkgo_create_cuda_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cu)
    target_compile_features("${TEST_TARGET_NAME}" PUBLIC cxx_std_14)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    target_compile_options(${TEST_TARGET_NAME}
        PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${GINKGO_CUDA_ARCH_FLAGS}>")
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_FAST_TESTS)
        target_compile_definitions(${TEST_TARGET_NAME} PRIVATE GINKGO_FAST_TESTS)
    endif()
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_hip_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")

    set_source_files_properties(${test_name}.hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
    set(GINKGO_TEST_HIP_DEFINES)
    if (GINKGO_FAST_TESTS)
        set(GINKGO_TEST_HIP_DEFINES -DGINKGO_FAST_TESTS)
    endif()

    # NOTE: With how HIP works, passing the flags `HIPCC_OPTIONS` etc. here
    # creates a redefinition of all flags. This creates some issues with `nvcc`,
    # but `clang` seems fine with the redefinitions.
    if (GINKGO_HIP_PLATFORM MATCHES "${HIP_PLATFORM_NVIDIA_REGEX}")
        hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
            # If `FindHIP.cmake`, namely `HIP_PARSE_HIPCC_OPTIONS` macro and
            # call gets fixed, uncomment this.
            HIPCC_OPTIONS ${GINKGO_TEST_HIP_DEFINES} # ${GINKGO_HIPCC_OPTIONS}
            # NVCC_OPTIONS  ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_NVCC_OPTIONS}
            # CLANG_OPTIONS ${GINKGO_TEST_HIP_DEFINES} ${GINKGO_HIP_CLANG_OPTIONS}
            )
    else() # hcc/clang
        hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS} ${GINKGO_TEST_HIP_DEFINES}
            NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
            CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS}
            )
    endif()

    # Let's use a normal compiler for linking
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)

    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        # Only `math` requires it so far, but it's much easier
        # to put these this way.
        ${GINKGO_HIP_THRUST_PATH}
        # Only `exception_helpers` requires these so far, but it's much easier
        # to put these this way.
        ${HIPBLAS_INCLUDE_DIRS}
        ${hiprand_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})

    if(BUILD_SHARED_LIBS)
        if (GINKGO_CHECK_CIRCULAR_DEPS)
            target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
        endif()
    endif()

    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name}
        COMMAND ${TEST_TARGET_NAME}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:ginkgo>")
endfunction(ginkgo_create_hip_test)
