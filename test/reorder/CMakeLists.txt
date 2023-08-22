ginkgo_create_common_test(amd DISABLE_EXECUTORS sycl)
if (GINKGO_HAVE_METIS)
    ginkgo_create_common_test(nested_dissection)
endif()
