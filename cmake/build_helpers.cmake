function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
            $<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
endfunction()

function(ginkgo_compile_features name)
    target_compile_features("${name}" PUBLIC cxx_std_14)
    if(GINKGO_WITH_CLANG_TIDY AND GINKGO_CLANG_TIDY_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_CLANG_TIDY "${GINKGO_CLANG_TIDY_PATH};-checks=*")
    endif()
    if(GINKGO_WITH_IWYU AND GINKGO_IWYU_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${GINKGO_IWYU_PATH})
    endif()
    # Set an appropriate SONAME
    set_property(TARGET "${name}" PROPERTY
        SOVERSION "${Ginkgo_VERSION}")
    if(GINKGO_CHANGED_SHARED_LIBRARY)
        # Put all shared libraries and corresponding imported libraries into the specified path
        set_property(TARGET "${name}" PROPERTY
            RUNTIME_OUTPUT_DIRECTORY "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
        set_property(TARGET "${name}" PROPERTY
            ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
        if(MSVC)
            # MSVC would create subfolder according to build_type. Ginkgo forces the output be the same whatever build_type is.
            foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
                string(TOUPPER ${CONFIG} CONFIG )
                set_property(TARGET "${name}" PROPERTY
                    RUNTIME_OUTPUT_DIRECTORY_${CONFIG} "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
                set_property(TARGET "${name}" PROPERTY
                    ARCHIVE_OUTPUT_DIRECTORY_${CONFIG} "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
            endforeach()
        endif()
        if(GINKGO_CHECK_PATH)
            ginkgo_check_shared_library("${CMAKE_SHARED_LIBRARY_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        endif()
    endif()

    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries("${name}" PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()

    set_target_properties("${name}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

function(ginkgo_check_headers target)
    # build object library used to "compile" the headers
    # add a proxy source file for each header in the target source list
    file(GLOB_RECURSE CUDA_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.cuh")
    file(GLOB_RECURSE HIP_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.hip.hpp")
    file(GLOB_RECURSE CXX_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.hpp")
    list(FILTER CXX_HEADERS EXCLUDE REGEX ".*\.hip\.hpp$")
    list(FILTER CXX_HEADERS EXCLUDE REGEX "^test.*")
    list(FILTER CUDA_HEADERS EXCLUDE REGEX "^test.*")
    list(FILTER HIP_HEADERS EXCLUDE REGEX "^test.*")

    set(SOURCES "")
    foreach(HEADER ${CUDA_HEADERS})
        set(HEADER_SOURCEFILE "${CMAKE_CURRENT_BINARY_DIR}/${HEADER}.cu")
        file(WRITE "${HEADER_SOURCEFILE}" "#include \"${HEADER}\"")
        list(APPEND SOURCES "${HEADER_SOURCEFILE}")
    endforeach()

    foreach(HEADER ${CXX_HEADERS})
        set(HEADER_SOURCEFILE "${CMAKE_CURRENT_BINARY_DIR}/${HEADER}.cpp")
        file(WRITE "${HEADER_SOURCEFILE}" "#include \"${HEADER}\"")
        list(APPEND SOURCES "${HEADER_SOURCEFILE}")
    endforeach()
    if (SOURCES)
        add_library(${target}_headers OBJECT ${SOURCES})
        target_link_libraries(${target}_headers PRIVATE ${target})
        target_include_directories(${target}_headers PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    set(HIP_SOURCES "")
    foreach(HEADER ${HIP_HEADERS})
        set(HEADER_SOURCEFILE "${CMAKE_CURRENT_BINARY_DIR}/${HEADER}.hip.cpp")
        file(WRITE "${HEADER_SOURCEFILE}" "#include \"${HEADER}\"")
        list(APPEND HIP_SOURCES "${HEADER_SOURCEFILE}")
    endforeach()
    if (HIP_SOURCES)
        set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
        hip_add_library(${target}_headers_hip ${HIP_SOURCES}) # the compiler options get set by linking to ginkgo_hip
        target_link_libraries(${target}_headers_hip PRIVATE ${target} roc::hipblas roc::hipsparse)
        target_include_directories(${target}_headers_hip
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${GINKGO_HIP_THRUST_PATH}"
            "${HIPBLAS_INCLUDE_DIRS}"
            "${HIPSPARSE_INCLUDE_DIRS}"
            "${ROCPRIM_INCLUDE_DIRS}")
    endif()
endfunction()

function(ginkgo_check_shared_library name)
    # Cygwin uses : not ; to split path
    if(CYGWIN)
        string(REPLACE ":" ";" ENV_PATH "$ENV{PATH}")
    else()
        set(ENV_PATH "$ENV{PATH}")
    endif()
    set(PATH_LIST ${ENV_PATH})
    set(PASSED_TEST FALSE)
    foreach(ITEM IN LISTS PATH_LIST)
        string(REPLACE "\\" "/" ITEM "${ITEM}")
        if("${ITEM}" STREQUAL "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
            set(PASSED_TEST TRUE)
            break()
        else()
            # If any path before this build, the path must not contain the ginkgo shared library
            find_file(EXISTING_DLL "${name}" PATHS "${ITEM}" NO_DEFAULT_PATH)
            if(NOT "${EXISTING_DLL}" STREQUAL "EXISTING_DLL-NOTFOUND")
                # clean the EXISTING_DLL before termination
                unset(EXISTING_DLL CACHE)
                message(FATAL_ERROR "Detect ${name} in ${ITEM} eariler than this build. "
                    "Please add ${GINKGO_WINDOWS_SHARED_LIBRARY_PATH} before other ginkgo path.")
            endif()
            # do not keep this variable in cache
            unset(EXISTING_DLL CACHE)
        endif()
    endforeach()
    if(NOT PASSED_TEST)
        # Did not find this build in the environment variable PATH
        message(FATAL_ERROR "Did not find this build in the environment variable PATH. "
            "Please add ${GINKGO_WINDOWS_SHARED_LIBRARY_PATH} into the environment variable PATH.")
    endif()
endfunction()

macro(ginkgo_modify_flags name)
    # add escape before "
    # the result var is ${name}_MODIFY
    string(REPLACE "\"" "\\\"" ${name}_MODIFY "${${name}}")
endmacro()

# Extract the clang version from a clang executable path
function(ginkgo_extract_clang_version CLANG_COMPILER GINKGO_CLANG_VERSION)
    set(CLANG_VERSION_PROG "#include <cstdio>\n"
        "int main() {printf(\"%d.%d.%d\", __clang_major__, __clang_minor__, __clang_patchlevel__)\;"
        "return 0\;}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp" ${CLANG_VERSION_PROG})
    execute_process(COMMAND ${CLANG_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver
        ERROR_VARIABLE CLANG_EXTRACT_VER_ERROR)
    execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver
        OUTPUT_VARIABLE FOUND_CLANG_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        )

    set (${GINKGO_CLANG_VERSION} "${FOUND_CLANG_VERSION}" PARENT_SCOPE)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver.cpp)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_clang_ver)
endfunction()

# Extract the DPC++ version
function(ginkgo_extract_dpcpp_version DPCPP_COMPILER GINKGO_DPCPP_VERSION)
    set(DPCPP_VERSION_PROG "#include <CL/sycl.hpp>\n#include <iostream>\n"
        "int main() {std::cout << __SYCL_COMPILER_VERSION << '\\n'\;"
        "return 0\;}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver.cpp" ${DPCPP_VERSION_PROG})
    execute_process(COMMAND ${DPCPP_COMPILER} ${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver.cpp
        -o ${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver
        ERROR_VARIABLE DPCPP_EXTRACT_VER_ERROR)
    execute_process(COMMAND ${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver
        OUTPUT_VARIABLE FOUND_DPCPP_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
        )

    set (${GINKGO_DPCPP_VERSION} "${FOUND_DPCPP_VERSION}" PARENT_SCOPE)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver.cpp)
    file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/extract_dpcpp_ver)
endfunction()
