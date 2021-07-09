execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --absolute-git-dir
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE Ginkgo_GIT_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(EXISTS "${Ginkgo_GIT_DIR}")
    set(ADD_HOOK FALSE)
    set(HOOK_LOCATION "${Ginkgo_SOURCE_DIR}/hooks/pre-commit")
    if(NOT EXISTS "${HOOK_LOCATION}")
        set(ADD_HOOK TRUE)
    else()
        # check if the correct hook is installed
        execute_process(COMMAND grep git-cmake-format.py "${HOOK_LOCATION}"
                        RESULT_VARIABLE res OUTPUT_QUIET)
        # return value =/= 0 means the pattern was not found
        if(NOT res EQUAL 0)
            set(ADD_HOOK TRUE)
        endif()
    endif()
    if(ADD_HOOK)
        configure_file(dummy_hook "${HOOK_LOCATION}" COPYONLY)
    endif()
endif()
