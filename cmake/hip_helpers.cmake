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

# This function checks if ROCm might not be detected correctly.
# ROCm < 5.7 has a faulty CMake setup that requires setting
# CMAKE_PREFIX_PATH=$ROCM_PATH/lib/cmake, otherwise HIP will not be detected.
function(ginkgo_check_hip_detection_issue)
    if(NOT CMAKE_HIP_COMPILER)
        ginkgo_find_hip_version()
        if(GINKGO_HIP_VERSION AND GINKGO_HIP_VERSION VERSION_LESS 5.7)
            message(
                WARNING
                "Could not find a HIP compiler, but HIP version ${GINKGO_HIP_VERSION} was detected through "
                "hipconfig. Try setting the environment variable CMAKE_PREFIX_PATH=$ROCM_PATH/lib/cmake, or "
                "update to ROCm >= 5.7."
            )
        endif()
    endif()
endfunction()
