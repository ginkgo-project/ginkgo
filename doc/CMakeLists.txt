add_subdirectory(_doxygen)

find_program(sphinx-build
    NAMES sphinx-build
    REQUIRED
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/conf.py @ONLY)
add_custom_target(sphinx-doc ALL
    COMMAND
    "${sphinx-build}" -M html ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    --conf-dir ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating sphinx documentation"
    VERBATIM
)
