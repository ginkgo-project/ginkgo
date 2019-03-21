# add_subdirectory(ex-3pt_stencil)
# add_subdirectory(ex-asynchronous_stopping_criterion)
# add_subdirectory(ex-custom_matrix_format)
# add_subdirectory(ex-ginkgo_overhead)
# add_subdirectory(ex-inverse_iteration)
# add_subdirectory(ex-minimal_solver_cuda)
# add_subdirectory(ex-papi_logging)
# add_subdirectory(ex-preconditioned_solver)
# add_subdirectory(ex-poisson_solver)
# add_subdirectory(ex-ranges)
# add_subdirectory(ex-simple_solver)
# add_subdirectory(ex-simple_solver_logging)

foreach(NUMBER RANGE 1 9)
  add_subdirectory(example-"$NUMBER")
endforeach()


