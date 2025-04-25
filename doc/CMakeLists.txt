add_subdirectory(doxygen)

find_program(sphinx-build NAMES sphinx-build REQUIRED)
find_program(doxysphinx NAMES doxysphinx REQUIRED)

# Copy to the sphinx sources in the build directory
# These have to be actual copies, since sphinx (or myst) can't
# resolve (symbolic) links to files outside the sphinx source
# directory
file(
    COPY index.md _static _templates developer-guide user-guide
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}
)

# Create symlinks for top-level files referenced by documentation
# Here links are fine, since those files are not part of the documentation,
# they are just referenced
file(
    CREATE_LINK ${Ginkgo_SOURCE_DIR}/LICENSE ${Ginkgo_BINARY_DIR}/LICENSE
    SYMBOLIC
)

if(DEFINED ENV{READTHEDOCS_OUTPUT})
    set(GINKGO_SPHINX_BUILD_DIR $ENV{READTHEDOCS_OUTPUT}/html)
else()
    set(GINKGO_SPHINX_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/html)
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
