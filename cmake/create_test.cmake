function(ginkgo_create_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test)

function(ginkgo_create_thread_test test_name)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest Threads::Threads ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_thread_test)

function(ginkgo_create_test_cpp_cuda_header test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        "${CUDA_INCLUDE_DIRS}"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_test_cpp_cuda_header)

function(ginkgo_create_cuda_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cu)
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})

    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_cuda_test)

function(ginkgo_create_hip_test_special_linkage test_name)
    # use gcc to compile but use hip to link
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")
    add_executable(${TEST_TARGET_NAME} ${test_name}.cpp)
    # Fix the missing metadata when building static library.
    if(GINKGO_HIP_PLATFORM MATCHES "hcc" AND NOT BUILD_SHARED_LIBS)
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES LINKER_LANGUAGE HIP)
    endif()
    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_hip_test_special_linkage)

function(ginkgo_create_hip_test test_name)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" TEST_TARGET_NAME "${REL_BINARY_DIR}/${test_name}")

    set_source_files_properties(${test_name}.hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

    if (HIP_VERSION GREATER_EQUAL "3.5")
        hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
            NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
            HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS}
            CLANG_OPTIONS ${GINKGO_HIP_CLANG_OPTIONS})
    else()
        hip_add_executable(${TEST_TARGET_NAME} ${test_name}.hip.cpp
            HIPCC_OPTIONS ${GINKGO_HIPCC_OPTIONS}
            NVCC_OPTIONS  ${GINKGO_HIP_NVCC_OPTIONS}
            HCC_OPTIONS ${GINKGO_HIP_HCC_OPTIONS})
    endif()

    # Let's really not use nvcc for linking here
    if (GINKGO_HIP_PLATFORM MATCHES "nvcc")
        set_target_properties(${TEST_TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
    endif()

    target_include_directories("${TEST_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        # Only `math` requires it so far, but it's much easier
        # to put these this way.
        ${GINKGO_HIP_THRUST_PATH}
        # Only `exception_helpers` requires thess so far, but it's much easier
        # to put these this way.
        ${HIPBLAS_INCLUDE_DIRS}
        ${HIPSPARSE_INCLUDE_DIRS}
        )
    set_target_properties(${TEST_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${test_name})

    # Pass in the `--amdgpu-target` flags if asked
    if(GINKGO_HIP_AMDGPU AND GINKGO_HIP_PLATFORM MATCHES "hcc")
        foreach(target ${GINKGO_HIP_AMDGPU})
            target_link_libraries(${TEST_TARGET_NAME} PRIVATE --amdgpu-target=${target})
        endforeach()
    endif()

    # GINKGO_RPATH_FOR_HIP needs to be populated before calling this for the linker to include
    # our libraries path into the executable's runpath.
    if(BUILD_SHARED_LIBS)
        target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_RPATH_FOR_HIP}")

        if (GINKGO_CHECK_CIRCULAR_DEPS)
            target_link_libraries(${TEST_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
        endif()
    endif()

    target_link_libraries(${TEST_TARGET_NAME} PRIVATE ginkgo GTest::Main GTest::GTest ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${test_name} COMMAND ${TEST_TARGET_NAME})
endfunction(ginkgo_create_hip_test)
