add_subdirectory(doxygen)

find_program(sphinx-build NAMES sphinx-build REQUIRED)
find_program(doxysphinx NAMES doxysphinx REQUIRED)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/conf.py
    @ONLY
)

# Copy to the sphinx sources in the build directory
# These have to be actual copies, since sphinx (or myst) can't
# resolve (symbolic) links to files outside the sphinx source
# directory
set(files_to_copy ${CMAKE_CURRENT_SOURCE_DIR}/index.md)
set(dirs_to_copy)
set(dirs _static _templates developer-guide user-guide)
foreach(dir IN LISTS dirs)
    file(GLOB_RECURSE globs LIST_DIRECTORIES true CONFIGURE_DEPENDS ${dir}/*)
    list(APPEND files_to_copy ${globs})

    cmake_path(ABSOLUTE_PATH dir OUTPUT_VARIABLE absolute_dir)
    list(APPEND dirs_to_copy ${absolute_dir})
endforeach()

foreach(in_source_file IN LISTS dirs_to_copy files_to_copy)
    string(
        REGEX REPLACE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}"
        in_binary_file
        ${in_source_file}
    )
    if(IS_DIRECTORY ${in_source_file})
        file(MAKE_DIRECTORY ${in_binary_file})
    else()
        file(COPY_FILE ${in_source_file} ${in_binary_file})
    endif()
endforeach()

# Create symlinks for top-level files referenced by documentation
# Here links are fine, since those files are not part of the documentation,
# they are just referenced
file(MAKE_DIRECTORY ${Ginkgo_BINARY_DIR}/test/test_install/)
file(
    CREATE_LINK ${Ginkgo_SOURCE_DIR}/LICENSE ${Ginkgo_BINARY_DIR}/LICENSE
    SYMBOLIC
)
file(
    CREATE_LINK
        ${Ginkgo_SOURCE_DIR}/test/test_install/CMakeLists.txt
        ${Ginkgo_BINARY_DIR}/test/test_install/CMakeLists.txt
    SYMBOLIC
)

if(DEFINED ENV{READTHEDOCS_OUTPUT})
    set(GINKGO_SPHINX_BUILD_DIR $ENV{READTHEDOCS_OUTPUT}/html)
else()
    set(GINKGO_SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/_build)
endif()

add_custom_target(
    doxysphinx
    ALL
    COMMAND
        "${doxysphinx}" build --doxygen_cwd ${CMAKE_CURRENT_BINARY_DIR}/doxygen/
        ${CMAKE_CURRENT_BINARY_DIR} ${GINKGO_SPHINX_BUILD_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/doxygen/Doxyfile
    DEPENDS doxygen
    COMMENT "Generating .rst files from doxygen documentation"
    VERBATIM
)

add_custom_target(
    sphinx-doc
    ALL
    COMMAND
        "${sphinx-build}" -j auto -b html --quiet ${CMAKE_CURRENT_BINARY_DIR}
        ${GINKGO_SPHINX_BUILD_DIR}
    DEPENDS doxysphinx
    COMMENT "Generating sphinx documentation"
    VERBATIM
)
