option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)

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
    custom-stopping-criterion
    ginkgo-overhead
    minimal-cuda-solver
    mixed-spmv
    mixed-multigrid-solver
    mixed-multigrid-preconditioned-solver
    multigrid-preconditioned-solver
    multigrid-preconditioned-solver-customized
    par-ilu-convergence
    performance-debugging
    preconditioner-export
    reordered-preconditioned-solver
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
            "GINKGO_WITH_CCACHE=ON which is known to cause issues with CUDA enabled "
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

if(GINKGO_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    set(executors)
    if(GINKGO_BUILD_REFERENCE)
        list(APPEND executors reference)
    endif()
    if(GINKGO_BUILD_CUDA)
        list(APPEND executors cuda)
    endif()
    if(GINKGO_BUILD_HIP)
        list(APPEND executors hip)
    endif()
    if(GINKGO_BUILD_SYCL)
        list(APPEND executors sycl)
    endif()
    if(GINKGO_BUILD_OMP)
        list(APPEND executors omp)
    endif()
    foreach(example IN LISTS EXAMPLES_EXEC_LIST)
        foreach(executor IN LISTS executors)
            add_test(NAME example_${example}_${executor}
                     COMMAND
                     ${Python3_EXECUTABLE}
                     "${CMAKE_CURRENT_SOURCE_DIR}/compare-output.py"
                     "${CMAKE_CURRENT_BINARY_DIR}/${example}"
                     "$<TARGET_FILE_DIR:ginkgo>"
                     "${CMAKE_CURRENT_SOURCE_DIR}/${example}/doc/results.dox"
                     "$<TARGET_FILE:${example}>"
                     "${executor}"
                     WORKING_DIRECTORY
                     "${CMAKE_CURRENT_SOURCE_DIR}/${example}")
        endforeach()
    endforeach()
endif()
