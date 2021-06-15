set(EXAMPLES_EXEC_LIST
    rm-simple-solver
    rm-simple-solver-separate
    )

set(EXAMPLES_LIST
    ${EXAMPLES_EXEC_LIST}
    )

foreach(example ${EXAMPLES_LIST})
    add_subdirectory(${example})
endforeach()
