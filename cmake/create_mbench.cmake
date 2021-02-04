function(ginkgo_create_mbench mbench_name)
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    file(RELATIVE_PATH REL_BINARY_DIR
        ${PROJECT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR})
    string(REPLACE "/" "_" MBENCH_TARGET_NAME "${REL_BINARY_DIR}/${mbench_name}")
    add_executable(${MBENCH_TARGET_NAME} ${mbench_name}.cpp)
    target_compile_features("${MBENCH_TARGET_NAME}" PUBLIC cxx_std_14)
    target_include_directories("${MBENCH_TARGET_NAME}"
        PRIVATE
        "$<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}>"
        )
    set_target_properties(${MBENCH_TARGET_NAME} PROPERTIES
        OUTPUT_NAME ${mbench_name})
    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries(${MBENCH_TARGET_NAME} PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()
    target_link_libraries(${MBENCH_TARGET_NAME} PRIVATE ginkgo GBench::GBench GBench::Main
        Threads::Threads ${ARGN})
    add_test(NAME ${REL_BINARY_DIR}/${mbench_name} COMMAND ${MBENCH_TARGET_NAME})
endfunction(ginkgo_create_mbench)
