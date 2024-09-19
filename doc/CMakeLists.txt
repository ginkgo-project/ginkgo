add_subdirectory(_doxygen)

find_program(sphinx-build NAMES sphinx-build REQUIRED)
find_program(doxysphinx NAMES doxysphinx REQUIRED)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/conf.py
    @ONLY
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
        "${doxysphinx}" build --doxygen_cwd
        ${CMAKE_CURRENT_SOURCE_DIR}/_build_doxygen ${CMAKE_CURRENT_SOURCE_DIR}
        ${GINKGO_SPHINX_BUILD_DIR} ${CMAKE_CURRENT_BINARY_DIR}/_doxygen/Doxyfile
    DEPENDS doxygen
    COMMENT "Generating .rst files from doxygen documentation"
    VERBATIM
)

file(
    CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/_static
        ${CMAKE_CURRENT_BINARY_DIR}/_static
    SYMBOLIC
)
file(
    CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/_templates
        ${CMAKE_CURRENT_BINARY_DIR}/_templates
    SYMBOLIC
)

add_custom_target(
    sphinx-doc
    ALL
    COMMAND
        "${sphinx-build}" -j auto -b html --quiet ${CMAKE_CURRENT_SOURCE_DIR}
        ${GINKGO_SPHINX_BUILD_DIR} --conf-dir ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS doxysphinx
    COMMENT "Generating sphinx documentation"
    VERBATIM
)
