cmake_minimum_required(VERSION 3.16)
find_program(sphinx-build
    NAMES sphinx-build
    REQUIRED
)

set(default_predefined_macros "GKO_HAVE_PAPI_SDE=1 GINKGO_BUILD_MPI=1")

add_custom_target(sphinx-doc ALL
    #DEPEND "${doxyfile}.stamp" Doxyfile.in ${in} ${in2}
    COMMAND "${sphinx-build}" -M html ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/build
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating sphinx documentation"
    VERBATIM
)
