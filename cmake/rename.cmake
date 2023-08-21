# Only for CACHE variable (option)
macro(gko_rename_cache deprecated actual type)
    if(DEFINED ${deprecated})
        if(DEFINED ${actual})
            message("actual ${actual} and deprecated ${deprecated}")
            if("${${actual}}" STREQUAL "${${deprecated}}")
                # They are the same, so only throw warning
                message(WARNING "${deprecated} was deprecated, please only use ${actual} instead.")
            else()
                # They are different
                set(${deprecated}_copy ${${deprecated}})
                unset(${deprecated} CACHE)
                message(FATAL_ERROR "Both ${deprecated} and ${actual} were specified, please use ${actual} instead.  "
                    "We remove ${deprecated}:${${deprecated}_copy} and keep ${actual}:${${actual}}")
            endif()
        else()
            # Only set `deprecated`, move it to `actual`.
            message(WARNING "${deprecated} was deprecated, please use ${actual} instead.  "
                "We copy ${${deprecated}} to ${actual} and unset ${deprecated}.")
            set(${actual} ${${deprecated}} CACHE ${type} "")
        endif()
        # We always unset the deprecated for easier next setup
        unset(${deprecated} CACHE)
    endif()
endmacro()