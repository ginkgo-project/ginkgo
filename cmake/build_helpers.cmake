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
        # Put all shared libraries and corresponding imported libraries into the specficed path
        set_property(TARGET "${name}" PROPERTY
            RUNTIME_OUTPUT_DIRECTORY "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
        set_property(TARGET "${name}" PROPERTY
            ARCHIVE_OUTPUT_DIRECTORY "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
        if(GINKGO_CHECK_PATH)
            ginkgo_check_shared_library("${CMAKE_SHARED_LIBRARY_PREFIX}${name}${CMAKE_SHARED_LIBRARY_SUFFIX}")
        endif()
    endif()
endfunction()

function(ginkgo_check_shared_library name)
    set(PATH_LIST $ENV{PATH})
    set(PASSED_TEST FALSE)
    foreach(ITEM ${PATH_LIST})
        string(REPLACE "\\" "/" ITEM "${ITEM}")
        if("${ITEM}" STREQUAL "${GINKGO_WINDOWS_SHARED_LIBRARY_PATH}")
            set(PASSED_TEST TRUE)
            break()
        else()
            # If any path before this build, the path must not contain the ginkgo shared library
            set(EXISTED_DLL "EXISTED_DLL-NOTFOUND")
            find_file(EXISTED_DLL "${name}" PATHS "${ITEM}" NO_DEFAULT_PATH)
            if(NOT "${EXISTED_DLL}" STREQUAL "EXISTED_DLL-NOTFOUND")
                message(FATAL_ERROR "Detect ${name} in ${ITEM} eariler than this build. "
                    "Please add ${GINKGO_WINDOWS_SHARED_LIBRARY_PATH} before other ginkgo path.")
            endif()
        endif()
    endforeach(ITEM)
    if(NOT PASSED_TEST)
        # Do not find this build in environment PATH
        message(FATAL_ERROR "Do not find this build in environment PATH. "
            "Please add ${GINKGO_WINDOWS_SHARED_LIBRARY_PATH} into environment PATH.")
    endif()
endfunction()