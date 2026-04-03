include(FindPackageHandleStandardArgs)

# Determine CUDA major version for version-specific cuDSS lookup
if(CMAKE_CUDA_COMPILER)
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        OUTPUT_VARIABLE _nvcc_version_output
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    string(REGEX MATCH "V([0-9]+)" _cuda_major "${_nvcc_version_output}")
    set(_cuda_major "${CMAKE_MATCH_1}")
endif()

find_path(
    CUDSS_INCLUDE_DIR
    NAMES cudss.h
    HINTS ${CUDSS_DIR} $ENV{CUDSS_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include
)

# Prefer version-specific cuDSS library matching CUDA major version
find_library(
    CUDSS_LIBRARY
    NAMES cudss
    HINTS ${CUDSS_DIR} $ENV{CUDSS_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES
        lib/x86_64-linux-gnu/libcudss/${_cuda_major}
        lib64/libcudss/${_cuda_major}
        lib
        lib64
)

find_package_handle_standard_args(
    cuDSS
    REQUIRED_VARS CUDSS_LIBRARY CUDSS_INCLUDE_DIR
)

if(cuDSS_FOUND AND NOT TARGET cuDSS::cuDSS)
    add_library(cuDSS::cuDSS UNKNOWN IMPORTED)
    set_target_properties(
        cuDSS::cuDSS
        PROPERTIES
            IMPORTED_LOCATION "${CUDSS_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDE_DIR}"
    )
endif()

mark_as_advanced(CUDSS_INCLUDE_DIR CUDSS_LIBRARY)
