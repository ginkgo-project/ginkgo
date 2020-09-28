option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES "Run the examples." OFF)

set(EXAMPLES_EXEC_LIST "adaptiveprecision-blockjacobi;custom-logger;ginkgo-overhead;ginkgo-ranges;ilu-preconditioned-solver;ir-ilu-preconditioned-solver;inverse-iteration;iterative-refinement;mixed-precision-ir;nine-pt-stencil-solver;poisson-solver;preconditioned-solver;simple-solver;three-pt-stencil-solver;")

set(EXAMPLES_LIST "${EXAMPLES_EXEC_LIST};custom-matrix-format;custom-stopping-criterion;minimal-cuda-solver;papi-logging;performance-debugging;preconditioner-export;simple-solver-logging")

if(GINKGO_BUILD_EXTLIB_EXAMPLE)
    set(EXAMPLES_LIST "${EXAMPLES_LIST};external-lib-interfacing")
endif()

foreach(example ${EXAMPLES_LIST})
    add_subdirectory(${example})
endforeach()

if(GINKGO_RUN_EXAMPLES)
    foreach(example ${EXAMPLES_LIST})
        add_custom_target("run-${example}"
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example} > ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example}.out
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${example})
    endforeach()
    foreach(example ${EXAMPLES_LIST})
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command "#!/bin/bash
diff <(sed -n '7,$p' ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example}.out | sed -E 's/([0-9]+.)//g') <(sed -n '6,$p' ${CMAKE_SOURCE_DIR}/examples/${example}/doc/results.dox | head -n -4 | sed -E 's/([0-9]+.)//g')")
        add_custom_target("validate-${example}"
            COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command && ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${example})
    endforeach()
    add_custom_target(run_all_examples)
    add_custom_target(validate_all_examples)
    foreach(run_ex ${EXAMPLES_EXEC_LIST})
        add_dependencies(run_all_examples "run-${run_ex}")
        add_dependencies(validate_all_examples "validate-${run_ex}")
    endforeach()
endif()
