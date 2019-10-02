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
    target_compile_features("${name}" PUBLIC cxx_std_11)
    if(GINKGO_WITH_CLANG_TIDY AND GINKGO_CLANG_TIDY_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_CLANG_TIDY "${GINKGO_CLANG_TIDY_PATH};-checks=*")
    endif()
    if(GINKGO_WITH_IWYU AND GINKGO_IWYU_PATH)
        set_property(TARGET "${name}" PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${GINKGO_IWYU_PATH})
    endif()
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

function(ginkgo_turn_windows_link lang from to)
    foreach(flag_var
        "CMAKE_${lang}_FLAGS" "CMAKE_${lang}_FLAGS_DEBUG" "CMAKE_${lang}_FLAGS_RELEASE"
        "CMAKE_${lang}_FLAGS_MINSIZEREL" "CMAKE_${lang}_FLAGS_RELWITHDEBINFO"
        )
        if(${flag_var} MATCHES "/${from}")
            string(REGEX REPLACE "/${from}" "/${to}" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/${from}")
        if(${flag_var} MATCHES "-${from}")
            string(REGEX REPLACE "-${from}" "-${to}" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "-${from}")
        set(${flag_var} "${${flag_var}}" CACHE STRING "" FORCE)
    endforeach()
endfunction()

macro(ginkgo_turn_to_windows_static lang)
    ginkgo_turn_windows_link(${lang} "MD" "MT")
endmacro()

macro(ginkgo_turn_to_windows_dynamic lang)
    ginkgo_turn_windows_link(${lang} "MT" "MD")
endmacro()