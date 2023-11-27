# Only for CACHE variable (option)
macro(gko_rename_cache deprecated actual type doc_string)
    if(DEFINED ${deprecated})
        if(DEFINED ${actual})
            message("actual ${actual} and deprecated ${deprecated}")
            if("${${actual}}" STREQUAL "${${deprecated}}")
                # They are the same, so only throw warning
                message(WARNING "${deprecated} was deprecated, please only use ${actual} instead. We removed the old variable.")
                unset(${deprecated} CACHE)
            else()
                # They are different
                message(FATAL_ERROR "Both ${deprecated} and ${actual} were specified differently, please only use ${actual} instead.")
            endif()
        else()
            # Only set `deprecated`, move it to `actual`.
            message(WARNING "${deprecated} was deprecated, please use ${actual} instead.  "
                "We copied ${${deprecated}} to ${actual} and removed the old variable.")
            set(${actual} ${${deprecated}} CACHE ${type} "${doc_string}")
            unset(${deprecated} CACHE)
        endif()
    endif()
endmacro()
