option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES "Run the examples." OFF)

function(setup_example_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} ginkgo)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
endfunction()


set(EXAMPLES_LIST "custom-logger;custom-matrix-format;custom-stopping-criterion;adaptiveprecision-blockjacobi;ginkgo-overhead;ginkgo-ranges;ilu-preconditioned-solver;ir-ilu-preconditioned-solver;inverse-iteration;iterative-refinement;minimal-cuda-solver;mixed-precision-ir;nine-pt-stencil-solver;papi-logging;performance-debugging;poisson-solver;preconditioned-solver;preconditioner-export;simple-solver;simple-solver-logging;three-pt-stencil-solver;")
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
endif()
