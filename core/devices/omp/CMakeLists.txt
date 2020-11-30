ginkgo_add_object_library(ginkgo_omp_device
  $<TARGET_OBJECTS:ginkgo_core_device> executor.cpp)
