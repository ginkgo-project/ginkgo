#.rst:
# FindPAPI
# -------
#
# Find PAPI, the Performance Application Programming Interface.
#
# This module only supports finding the PAPI C library and headers.
# This module optionally supports searching for PAPI components by
# using the standard PAPI functions. To find which components are
# activated on your system use the command `papi_component_avail`.
#
# Imported targets
# ^^^^^^^^^^^^^^^^
#
# This module defines the following :prop_tgt:`IMPORTED` target:
#
# ``PAPI::PAPI``
#   The PAPI library, if found.
#
# Result variables
# ^^^^^^^^^^^^^^^^
#
# This module will set the following variables in your project:
#
# ``PAPI_INCLUDE_DIRS``
#   where to find papi.h
#
# ``PAPI_LIBRARIES``
#   the libraries to link against in order to use PAPI.
#
# ``PAPI_FOUND``
#   If false, do not try to use PAPI.
#
# ``PAPI_VERSION_STRING``
#   the version of the PAPI library found
#
# ``PAPI_<COMPONENT>_FOUND``
#   If false, the specified PAPI component was not found

find_path(PAPI_INCLUDE_DIR NAMES papi.h)
mark_as_advanced(PAPI_INCLUDE_DIR)

if(NOT PAPI_LIBRARY)
    find_library(PAPI_LIBRARY_RELEASE NAMES
        papi
        papi64
        )
    mark_as_advanced(PAPI_LIBRARY_RELEASE)

    find_library(PAPI_LIBRARY_DEBUG NAMES
        papid
        papi-d
        )
    mark_as_advanced(PAPI_LIBRARY_DEBUG)

    include(SelectLibraryConfigurations)
    select_library_configurations(PAPI)
endif()

if(PAPI_INCLUDE_DIR)
    if(EXISTS "${PAPI_INCLUDE_DIR}/papi.h")
        file(STRINGS "${PAPI_INCLUDE_DIR}/papi.h" papi_version_str REGEX "^#define[\t ]+PAPI_VERSION[\t ]+.*")

        string(REGEX REPLACE "^#define[\t ]+PAPI_VERSION[\t ]+PAPI_VERSION_NUMBER[(]+([0-9,]*)[)]+" "\\1" PAPI_VERSION_STRING "${papi_version_str}")
        string(REPLACE "," "." PAPI_VERSION_STRING "${PAPI_VERSION_STRING}")
        unset(papi_version_str)
    endif()

    if (PAPI_LIBRARY)
        # find the components
       foreach(component IN LISTS PAPI_FIND_COMPONENTS)
           file(WRITE "${CMAKE_BINARY_DIR}/papi_${component}_detect.c"
             "
             #include <papi.h>
             int main() {
              int retval;
              retval = PAPI_library_init(PAPI_VER_CURRENT);
                if (retval != PAPI_VER_CURRENT && retval > 0)
                 return -1;
                if (PAPI_get_component_index(\"${component}\") < 0)
                 return 0;
                return 1;
             }
            ")
        try_run(PAPI_${component}_FOUND
            gko_result_unused
            "${CMAKE_BINARY_DIR}"
            "${CMAKE_BINARY_DIR}/papi_${component}_detect.c"
            LINK_LIBRARIES ${PAPI_LIBRARY}
            )

        if (NOT PAPI_${component}_FOUND EQUAL 1)
            unset(PAPI_${component}_FOUND)
        endif()
        endforeach()
    endif()
endif()


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PAPI
                                  REQUIRED_VARS PAPI_LIBRARY PAPI_INCLUDE_DIR
                                  VERSION_VAR PAPI_VERSION_STRING
                                  HANDLE_COMPONENTS)

if(PAPI_FOUND)
    set(PAPI_LIBRARIES ${PAPI_LIBRARY})
    set(PAPI_INCLUDE_DIRS ${PAPI_INCLUDE_DIR})
    unset(PAPI_LIBRARY)
    unset(PAPI_INCLUDE_DIR)

    if(NOT TARGET PAPI::PAPI)
        add_library(PAPI::PAPI UNKNOWN IMPORTED)
        set_target_properties(PAPI::PAPI PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PAPI_INCLUDE_DIRS}")

        if(EXISTS "${PAPI_LIBRARIES}")
            set_target_properties(PAPI::PAPI PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION "${PAPI_LIBRARIES}")
        endif()
        if(PAPI_LIBRARY_RELEASE)
            set_property(TARGET PAPI::PAPI APPEND PROPERTY
                IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(PAPI::PAPI PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION_RELEASE "${PAPI_LIBRARY_RELEASE}")
            unset(PAPI_LIBRARY_RELEASE)
        endif()
        if(PAPI_LIBRARY_DEBUG)
            set_property(TARGET PAPI::PAPI APPEND PROPERTY
                IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(PAPI::PAPI PROPERTIES
                IMPORTED_LINK_INTERFACE_LANGUAGES "C"
                IMPORTED_LOCATION_DEBUG "${PAPI_LIBRARY_DEBUG}")
            unset(PAPI_LIBRARY_DEBUG)
        endif()
    endif()
endif()
