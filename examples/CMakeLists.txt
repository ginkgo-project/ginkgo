option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)

set(EXAMPLES_EXEC_LIST
    adaptiveprecision-blockjacobi
    batched-solver
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

find_package(Kokkos 4.1.00 QUIET)
if(Kokkos_FOUND)
    list(APPEND EXAMPLES_LIST kokkos-assembly)
else()
    message(STATUS "No Kokkos found, disabling examples with Kokkos assembly.")
endif()

set(GKO_FILE_CONFIG_ENABLED 0)
find_package(nlohmann_json 3.9.1 QUIET)
if(nlohmann_json_FOUND)
    set(GKO_FILE_CONFIG_ENABLED 1)
    list(APPEND EXAMPLES_LIST file-config-solver)
else()
    message(STATUS "No nlohmann-json found, disabling file-config example")
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
    # OpenMP support is needed for some host kernels
    if(GINKGO_BUILD_CUDA AND GINKGO_BUILD_OMP)
        list(APPEND executors cuda)
    endif()
    if(GINKGO_BUILD_HIP AND GINKGO_BUILD_OMP)
        list(APPEND executors hip)
    endif()
    # DPC++ support is incomplete
    #if(GINKGO_BUILD_SYCL)
    #    list(APPEND executors dpcpp)
    #endif()
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


if(GKO_FILE_CONFIG_ENABLED)
    file(GLOB config_list RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" file-config-solver/config/*.json)
    foreach(config IN LISTS config_list)
        get_filename_component(config_name "${config}" NAME_WE)
        foreach(executor IN LISTS executors)
            add_test(NAME example_file-config-solver_${config_name}_${executor} 
                     COMMAND 
                     "$<TARGET_FILE:file-config-solver>"
                     "${executor}" "${CMAKE_CURRENT_SOURCE_DIR}/file-config-solver/config/${config_name}.json"
                     "${CMAKE_CURRENT_SOURCE_DIR}/file-config-solver/data/A.mtx"
                     WORKING_DIRECTORY
                     "$<TARGET_FILE_DIR:ginkgo>")
        endforeach()    
    endforeach()
endif()
        
endif()
