if (GINKGO_HAVE_METIS)
    ginkgo_create_common_test(nested_dissection DISABLE_EXECUTORS cuda hip dpcpp)
endif()