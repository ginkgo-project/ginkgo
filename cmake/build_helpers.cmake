set(GINKGO_LIBRARY_PATH "${PROJECT_BINARY_DIR}/lib")

function(ginkgo_default_includes name)
    # set include path depending on used interface
    target_include_directories("${name}"
        PUBLIC
            $<BUILD_INTERFACE:${Ginkgo_BINARY_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${Ginkgo_SOURCE_DIR}>
            $<INSTALL_INTERFACE:include>
        )
    if(GINKGO_HAVE_HWLOC)
      target_include_directories("${name}"
        PUBLIC
        $<BUILD_INTERFACE:${HWLOC_INCLUDE_DIRS}>
        )
    endif()
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
    # Put all shared libraries and corresponding imported libraries into the specified path
    set_property(TARGET "${name}" PROPERTY
        RUNTIME_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}")
    set_property(TARGET "${name}" PROPERTY
        ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}")
    set_property(TARGET "${name}" PROPERTY
        LIBRARY_OUTPUT_DIRECTORY "${GINKGO_LIBRARY_PATH}")

    if (GINKGO_CHECK_CIRCULAR_DEPS)
        target_link_libraries("${name}" PRIVATE "${GINKGO_CIRCULAR_DEPS_FLAGS}")
    endif()

    set_target_properties("${name}" PROPERTIES POSITION_INDEPENDENT_CODE ON)
endfunction()

function(ginkgo_check_headers target defines)
    # build object library used to "compile" the headers
    # add a proxy source file for each header in the target source list
    file(GLOB_RECURSE CUDA_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.cuh")
    file(GLOB_RECURSE HIP_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.hip.hpp")
    file(GLOB_RECURSE CXX_HEADERS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" CONFIGURE_DEPENDS "*.hpp")
    list(FILTER CXX_HEADERS EXCLUDE REGEX ".*\.hip\.hpp$")
    list(FILTER CXX_HEADERS EXCLUDE REGEX "^test.*")
    list(FILTER CXX_HEADERS EXCLUDE REGEX "^base/kernel_launch.*")
    list(FILTER CUDA_HEADERS EXCLUDE REGEX "^test.*")
    list(FILTER CUDA_HEADERS EXCLUDE REGEX "^base/kernel_launch.*")
    list(FILTER HIP_HEADERS EXCLUDE REGEX "^test.*")
    list(FILTER HIP_HEADERS EXCLUDE REGEX "^base/kernel_launch.*")

    set(SOURCES "")
    # if we have any CUDA files in there, compile everything as CUDA
    if(CUDA_HEADERS)
        set(CUDA_HEADERS ${CUDA_HEADERS} ${CXX_HEADERS})
        set(CXX_HEADERS "")
        if (HIP_HEADERS)
            message(FATAL_ERROR "Mixing CUDA and HIP files in header check")
        endif()
    endif()
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
    if(SOURCES)
        add_library(${target}_headers OBJECT ${SOURCES})
        target_link_libraries(${target}_headers PRIVATE ${target})
        target_include_directories(${target}_headers PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
        if(defines)
            target_compile_definitions(${target}_headers PRIVATE ${defines})
        endif()
    endif()

    set(HIP_SOURCES "")
    foreach(HEADER ${HIP_HEADERS})
        set(HEADER_SOURCEFILE "${CMAKE_CURRENT_BINARY_DIR}/${HEADER}.hip.cpp")
        file(WRITE "${HEADER_SOURCEFILE}" "#include \"${HEADER}\"")
        list(APPEND HIP_SOURCES "${HEADER_SOURCEFILE}")
    endforeach()
    if(HIP_SOURCES)
        set_source_files_properties(${HIP_SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
        hip_add_library(${target}_headers_hip ${HIP_SOURCES}) # the compiler options get set by linking to ginkgo_hip
        target_link_libraries(${target}_headers_hip PRIVATE ${target} roc::hipblas roc::hipsparse hip::hiprand roc::rocrand)
        target_include_directories(${target}_headers_hip
            PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            "${GINKGO_HIP_THRUST_PATH}"
            "${HIPBLAS_INCLUDE_DIRS}"
            "${hiprand_INCLUDE_DIRS}"
            "${HIPSPARSE_INCLUDE_DIRS}"
            "${ROCPRIM_INCLUDE_DIRS}")
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
function(ginkgo_extract_dpcpp_version DPCPP_COMPILER GINKGO_DPCPP_VERSION MACRO_VAR)
    set(DPCPP_VERSION_PROG "#include <CL/sycl.hpp>\n#include <iostream>\n"
        "int main() {std::cout << ${MACRO_VAR} << '\\n'\;"
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
