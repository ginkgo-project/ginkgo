add_subdirectory(doxygen)

find_program(sphinx-build NAMES sphinx-build REQUIRED)
find_program(doxysphinx NAMES doxysphinx REQUIRED)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/conf.py
    @ONLY
)

# Creating symlinks to the sphinx sources in the build directory
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
file(
    CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/developer-guide
        ${CMAKE_CURRENT_BINARY_DIR}/developer-guide
    SYMBOLIC
)
file(
    CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/user-guide
        ${CMAKE_CURRENT_BINARY_DIR}/user-guide
    SYMBOLIC
)
file(
    CREATE_LINK
        ${CMAKE_CURRENT_SOURCE_DIR}/index.md
        ${CMAKE_CURRENT_BINARY_DIR}/index.md
    SYMBOLIC
)

# Create symlinks for top-level files referenced by documentation
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
