# IntelSYCL for dpcpp and icpx if the config is existed and cmake reaches the requirement
if(CMAKE_CXX_COMPILER MATCHES "dpcpp|icpx")
    if(CMAKE_HOST_WIN32 AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
        find_package(IntelSYCL QUIET)
    elseif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20.5)
        find_package(IntelSYCL QUIET)
    endif()
endif()
# If we do not have the config from compiler, try to set components to make it work.
if(NOT COMMAND add_sycl_to_target) 
    if(NOT DEFINED SYCL_FLAGS)
        set(SYCL_FLAGS "-fsycl" CACHE STRING "SYCL flags for compiler")
    endif()
endif()

# Provide a uniform way for those package without add_sycl_to_target
function(gko_add_sycl_to_target)
    if(COMMAND add_sycl_to_target)
        add_sycl_to_target(${ARGN})
        return()
    endif()
    # We handle them by adding SYCL_FLAGS to compile and link to the target
    set(one_value_args TARGET)
    set(multi_value_args SOURCES)
    cmake_parse_arguments(SYCL
        ""
        "${one_value_args}"
        "${multi_value_args}"
        ${ARGN})
    target_compile_options(${SYCL_TARGET} PRIVATE "${SYCL_FLAGS}")
    target_link_options(${SYCL_TARGET} PRIVATE "${SYCL_FLAGS}")
endfunction()

