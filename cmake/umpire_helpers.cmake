# Helpers for Ginkgo's optional Umpire support (GINKGO_BUILD_UMPIRE).

# Detects Umpire and sets GINKGO_HAVE_UMPIRE in the caller's scope. Umpire is
# never fetched: it comes from a parent project or from an installation.
function(ginkgo_find_umpire)
    set(GINKGO_HAVE_UMPIRE 0)
    # The version is unknown for a parent-provided target, so default to C++20.
    set(GINKGO_UMPIRE_CXX_STANDARD cxx_std_20)
    if(GINKGO_BUILD_UMPIRE)
        if(TARGET umpire::umpire OR TARGET umpire)
            # Reuse an Umpire target added by a parent project; a second copy
            # would fail with a duplicate-target error.
            message(
                STATUS
                "Using the Umpire target provided by the parent project"
            )
            set(GINKGO_HAVE_UMPIRE 1)
        else()
            if(CMAKE_VERSION VERSION_LESS 3.23)
                message(
                    FATAL_ERROR
                    "GINKGO_BUILD_UMPIRE requires CMake >= 3.23 (Umpire's own minimum); "
                    "found ${CMAKE_VERSION}."
                )
            endif()
            # find_package(umpire) reruns BLT's CUDA setup (umpire-config.cmake ->
            # campConfig.cmake -> BLTSetupTargets.cmake -> BLTSetupCUDA.cmake),
            # which fails if CUDA is enabled and CMAKE_CUDA_HOST_COMPILER is empty.
            # Set it for the duration of find_package only, then restore it.
            set(GINKGO_UMPIRE_RESTORE_CUDA_HOST_COMPILER OFF)
            get_property(
                GINKGO_UMPIRE_LANGUAGES
                GLOBAL
                PROPERTY ENABLED_LANGUAGES
            )
            if(
                "CUDA" IN_LIST GINKGO_UMPIRE_LANGUAGES
                AND NOT CMAKE_CUDA_HOST_COMPILER
            )
                set(GINKGO_UMPIRE_SAVED_CUDA_HOST_COMPILER
                    "${CMAKE_CUDA_HOST_COMPILER}"
                )
                set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}")
                set(GINKGO_UMPIRE_RESTORE_CUDA_HOST_COMPILER ON)
            endif()
            # camp's BLT macros rerun the legacy FindCUDA, which defaults to
            # /usr/local/cuda and asserts that it exists
            if(
                "CUDA" IN_LIST GINKGO_UMPIRE_LANGUAGES
                AND CUDAToolkit_LIBRARY_ROOT
                AND NOT CUDA_TOOLKIT_ROOT_DIR
            )
                set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_LIBRARY_ROOT}")
            endif()
            find_package(umpire QUIET)
            if(GINKGO_UMPIRE_RESTORE_CUDA_HOST_COMPILER)
                set(CMAKE_CUDA_HOST_COMPILER
                    "${GINKGO_UMPIRE_SAVED_CUDA_HOST_COMPILER}"
                )
                unset(GINKGO_UMPIRE_SAVED_CUDA_HOST_COMPILER)
            endif()
            unset(GINKGO_UMPIRE_RESTORE_CUDA_HOST_COMPILER)
            unset(GINKGO_UMPIRE_LANGUAGES)
            if(NOT umpire_FOUND)
                message(
                    FATAL_ERROR
                    "GINKGO_BUILD_UMPIRE=ON but no Umpire installation was found. "
                    "Point -DCMAKE_PREFIX_PATH or -Dumpire_DIR at an Umpire "
                    "installation of version 2025.09 or newer, ideally the same "
                    "one your application uses, or set GINKGO_BUILD_UMPIRE=OFF."
                )
            endif()
            message(STATUS "Found Umpire ${umpire_VERSION}")
            # Only isAllocator()/getAllocator() are used, not the
            # >= v2026.07.0 tryGetAllocator()/destroyAllocator().
            if(umpire_VERSION VERSION_LESS 2025.09)
                message(
                    FATAL_ERROR
                    "Found Umpire ${umpire_VERSION}, but Ginkgo requires Umpire >= 2025.09."
                )
            endif()
            # Umpire >= 2026.07 requires C++20; 2025.09 to 2025.12 need C++17.
            if(umpire_VERSION VERSION_LESS 2026.07)
                set(GINKGO_UMPIRE_CXX_STANDARD cxx_std_17)
            endif()
            set(GINKGO_HAVE_UMPIRE 1)
        endif()
        if(GINKGO_HAVE_UMPIRE)
            message(
                STATUS
                "Umpire glue C++ standard: ${GINKGO_UMPIRE_CXX_STANDARD}"
            )
        endif()
    endif()
    set(GINKGO_HAVE_UMPIRE ${GINKGO_HAVE_UMPIRE} PARENT_SCOPE)
    set(GINKGO_UMPIRE_CXX_STANDARD ${GINKGO_UMPIRE_CXX_STANDARD} PARENT_SCOPE)
endfunction()

# Sets GINKGO_UMPIRE_TARGET in the caller's scope. Must be called after
# add_subdirectory(third_party), once the Umpire target is known.
function(ginkgo_resolve_umpire_target)
    if(GINKGO_HAVE_UMPIRE)
        # v2026.07.0 renamed the export to umpire::umpire and deprecated the bare
        # target; older installs only provide the bare one.
        if(TARGET umpire::umpire)
            set(GINKGO_UMPIRE_TARGET umpire::umpire)
        elseif(TARGET umpire)
            set(GINKGO_UMPIRE_TARGET umpire)
        else()
            message(
                FATAL_ERROR
                "Umpire was requested but no umpire target exists."
            )
        endif()
        message(STATUS "Umpire link target: ${GINKGO_UMPIRE_TARGET}")
        set(GINKGO_UMPIRE_TARGET ${GINKGO_UMPIRE_TARGET} PARENT_SCOPE)
    endif()
endfunction()

# Adds the Umpire allocator glue (or its stub) to the Ginkgo core target.
#
# @param target  the name of the Ginkgo core target
function(ginkgo_add_umpire_to_target target)
    if(GINKGO_HAVE_UMPIRE)
        if(NOT BUILD_SHARED_LIBS)
            # Static builds with Umpire are untested, so they are rejected
            # rather than left to fail in confusing ways.
            message(
                FATAL_ERROR
                "GINKGO_BUILD_UMPIRE=ON is not currently supported together "
                "with BUILD_SHARED_LIBS=OFF: static builds with Umpire are "
                "untested. Build Ginkgo with BUILD_SHARED_LIBS=ON "
                "(the default) to use Umpire, or set GINKGO_BUILD_UMPIRE=OFF "
                "for a static build."
            )
        endif()
        # Umpire >= v2026.07.0 requires C++20 while the rest of Ginkgo is C++17.
        # Isolating the glue in its own object library keeps the standard bump to
        # this one translation unit; linking PRIVATE keeps it out of Ginkgo's
        # INTERFACE.
        add_library(ginkgo_umpire OBJECT base/umpire_allocator.cpp)
        target_compile_features(
            ginkgo_umpire
            PRIVATE ${GINKGO_UMPIRE_CXX_STANDARD}
        )
        target_link_libraries(ginkgo_umpire PRIVATE ${GINKGO_UMPIRE_TARGET})
        ginkgo_default_includes(ginkgo_umpire)
        set_target_properties(
            ginkgo_umpire
            PROPERTIES POSITION_INDEPENDENT_CODE ON
        )
        target_link_libraries(${target} PRIVATE ginkgo_umpire)
    else()
        target_sources(${target} PRIVATE device_hooks/umpire_hooks.cpp)
    endif()
endfunction()
