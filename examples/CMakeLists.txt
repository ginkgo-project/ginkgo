option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES "Run the examples." OFF)

function(setup_example_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} ginkgo)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
endfunction()

set(EXAMPLES_EXEC_LIST "adaptiveprecision-blockjacobi;custom-logger;ginkgo-overhead;ginkgo-ranges;ilu-preconditioned-solver;ir-ilu-preconditioned-solver;inverse-iteration;iterative-refinement;mixed-precision-ir;nine-pt-stencil-solver;poisson-solver;preconditioned-solver;simple-solver;three-pt-stencil-solver;")

set(EXAMPLES_LIST "${EXAMPLES_EXEC_LIST};custom-matrix-format;custom-stopping-criterion;minimal-cuda-solver;papi-logging;performance-debugging;preconditioner-export;simple-solver-logging")

if(GINKGO_BUILD_EXTLIB_EXAMPLE)
    set(EXAMPLES_LIST "${EXAMPLES_LIST};external-lib-interfacing")
endif()

foreach(example ${EXAMPLES_LIST})
    add_subdirectory(${example})
endforeach()

set(run_list "")
if(GINKGO_RUN_EXAMPLES)
    foreach(example ${EXAMPLES_LIST})
        add_custom_target("run-${example}"
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example} > ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example}.out
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${example})
    endforeach()
    foreach(example ${EXAMPLES_LIST})
        file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command "#!/bin/bash
diff <(sed -n '7,$p' ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example}.out | sed -E 's/([0-9]+.)//g') <(sed -n '6,$p' ${CMAKE_SOURCE_DIR}/examples/${example}/doc/results.dox | head -n -4 | sed -E 's/([0-9]+.)//g')")
        add_custom_target("diff-${example}"
            COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command && ${CMAKE_CURRENT_BINARY_DIR}/${example}/diff-command
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${example})
    endforeach()
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/run_all_examples.sh
        "NUM_EXAMPLES=$(wc -l < example-targets.txt)\n"
        "for i in $(seq 1 $NUM_EXAMPLES)\n"
        "do\n"
        "  target_name=$(head -n $i example-targets.txt | tail -1)\n"
        "  ex_target=run-\$\{target_name\}\n"
        "  cmake --build ${CMAKE_BINARY_DIR} --target $ex_target\n"
        "done\n")
    add_custom_target("run_all_examples"
        COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/run_all_examples.sh && ${CMAKE_CURRENT_BINARY_DIR}/run_all_examples.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/validate_all_examples.sh
        "NUM_EXAMPLES=$(wc -l < example-targets.txt)\n"
        "for i in $(seq 1 $NUM_EXAMPLES)\n"
        "do\n"
        "  target_name=$(head -n $i example-targets.txt | tail -1)\n"
        "  ex_target=diff-\$\{target_name\}\n"
        "  cmake --build ${CMAKE_BINARY_DIR} --target $ex_target\n"
        "done\n")
    add_custom_target("validate_all_examples"
        COMMAND chmod +x ${CMAKE_CURRENT_BINARY_DIR}/validate_all_examples.sh && ${CMAKE_CURRENT_BINARY_DIR}/validate_all_examples.sh
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()

set(OUT_FILE "${CMAKE_BINARY_DIR}/examples/example-targets.txt")
file(WRITE ${OUT_FILE} "")
foreach(example ${EXAMPLES_EXEC_LIST})
    file(APPEND ${OUT_FILE} "${example}\n")
endforeach()
