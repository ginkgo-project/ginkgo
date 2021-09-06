option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES " Compile run and validation targets for the examples." ON)

set(EXAMPLES_EXEC_LIST
    adaptiveprecision-blockjacobi
    batched-solver
    batched-solver-from-files
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
    custom-stopping-criterion
    ginkgo-overhead
    minimal-cuda-solver
    mixed-spmv
    mixed-multigrid-solver
    multigrid-preconditioned-solver
    par-ilu-convergence
    performance-debugging
    preconditioner-export
    simple-solver-logging)

if(GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
    list(APPEND EXAMPLES_LIST custom-matrix-format)
endif()

if(GINKGO_BUILD_EXTLIB_EXAMPLE)
    list(APPEND EXAMPLES_LIST external-lib-interfacing)
endif()

find_package(OpenCV QUIET)
if(OpenCV_FOUND)
    list(APPEND EXAMPLES_LIST heat-equation schroedinger-splitting)
else()
    message(STATUS "No OpenCV found, disabling examples with video output")
endif()

if(GINKGO_HAVE_PAPI_SDE)
    list(APPEND EXAMPLES_LIST papi-logging)
endif()

if(GINKGO_BUILD_MPI)
    list(APPEND EXAMPLES_LIST distributed-solver)
endif()

find_package(Kokkos QUIET)
if(Kokkos_FOUND)
    if(GINKGO_WITH_CCACHE)
        message(WARNING "The CMAKE_CXX_COMPILER_LAUNCHER is set due to "
            "GINKGO_WITH_CCACHE=ON which is known to casue issues with CUDA enabled "
            "Kokkos (https://github.com/kokkos/kokkos/issues/4821) including compilation "
            "failures. This can be prevented by setting GINKGO_WITH_CCACHE=OFF.")
    endif()
    list(APPEND EXAMPLES_LIST kokkos_assembly)
else()
    message(STATUS "No Kokkos found, disabling examples with Kokkos assembly.")
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
diff <(sed -n '8,$p' ${example_path}/${example}.out | sed -E 's/([^a-z,\":\\s\\)\\(\\{\\}_]+)//g') <(sed -n '6,$p' ${CMAKE_SOURCE_DIR}/examples/${example}/doc/results.dox | head -n -4 | sed -E 's/([^a-z,\":\\s\\)\\(\\{\\}_]+)//g')")
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
