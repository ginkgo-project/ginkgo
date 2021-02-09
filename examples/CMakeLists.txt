option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES " Compile run and validation targets for the examples." ON)

set(EXAMPLES_EXEC_LIST
    adaptiveprecision-blockjacobi
    cb-gmres
    custom-logger
    ginkgo-ranges
    ilu-preconditioned-solver
    ir-ilu-preconditioned-solver
    inverse-iteration
    iterative-refinement
    mixed-precision-ir
    nine-pt-stencil-solver
    poisson-solver
    preconditioned-solver
    simple-solver
    three-pt-stencil-solver)

set(EXAMPLES_LIST
    ${EXAMPLES_EXEC_LIST}
    custom-matrix-format
    custom-stopping-criterion
    ginkgo-overhead
    heat-equation
    minimal-cuda-solver
    papi-logging
    par-ilu-convergence
    performance-debugging
    preconditioner-export
    simple-solver-logging)

if(GINKGO_BUILD_EXTLIB_EXAMPLE)
    list(APPEND EXAMPLES_LIST external-lib-interfacing)
endif()

foreach(example ${EXAMPLES_LIST})
    add_subdirectory(${example})
endforeach()

if(GINKGO_RUN_EXAMPLES)
    foreach(example ${EXAMPLES_LIST})
        set(example_path "${CMAKE_CURRENT_BINARY_DIR}/${example}")
        file(WRITE ${example_path}/target-wrapper.sh "${example_path}/${example} \$\{EX_ARG\}")
        add_custom_target("run-${example}"
            COMMAND chmod +x ${example_path}/target-wrapper.sh && ${example_path}/target-wrapper.sh > ${example_path}/${example}.out
            WORKING_DIRECTORY ${example_path})
        file(WRITE ${example_path}/diff-command "#!/bin/bash
diff <(sed -n '7,$p' ${example_path}/${example}.out | sed -E 's/([^a-z,\":\\s\\)\\(\\{\\}_]+)//g') <(sed -n '6,$p' ${CMAKE_SOURCE_DIR}/examples/${example}/doc/results.dox | head -n -4 | sed -E 's/([^a-z,\":\\s\\)\\(\\{\\}_]+)//g')")
        add_custom_target("validate-${example}"
            COMMAND chmod +x ${example_path}/diff-command && ${example_path}/diff-command
            WORKING_DIRECTORY ${example_path})
    endforeach()

    add_custom_target(run_all_examples)
    add_custom_target(validate_all_examples)
    foreach(run_ex ${EXAMPLES_EXEC_LIST})
        add_dependencies(run_all_examples "run-${run_ex}")
        add_dependencies(validate_all_examples "validate-${run_ex}")
    endforeach()
endif()
