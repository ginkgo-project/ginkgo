option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the example-6 with deal.II, you need to link the deal.II library." OFF)
foreach(NUMBER RANGE 1 9)
  add_subdirectory(example-${NUMBER})
endforeach()
