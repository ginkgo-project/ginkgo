function(ginkgo_find_hip_version)
    find_program(
        HIP_HIPCONFIG_EXECUTABLE
        NAMES hipconfig
        PATHS "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
    )
    if(NOT HIP_HIPCONFIG_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HIPCONFIG_EXECUTABLE hipconfig)
    endif()

    execute_process(
        COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --version
        OUTPUT_VARIABLE GINKGO_HIP_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )
    set(GINKGO_HIP_VERSION ${GINKGO_HIP_VERSION} PARENT_SCOPE)
endfunction()
