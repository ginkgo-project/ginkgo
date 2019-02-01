ginkgo_create_test(convergence)
if (GKO_HAVE_PAPI_SDE)
	ginkgo_create_test(papi)
endif()
