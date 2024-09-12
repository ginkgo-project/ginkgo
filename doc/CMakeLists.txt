add_subdirectory(_doxygen)

find_program(sphinx-build
    NAMES sphinx-build
    REQUIRED
)

add_custom_target(sphinx-doc ALL
    #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
    COMMAND "${sphinx-build}" -M html ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating sphinx documentation"
    VERBATIM
)
