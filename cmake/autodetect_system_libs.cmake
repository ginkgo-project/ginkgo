if (NOT DEFINED GINKGO_BUILD_PAPI_SDE)
    find_package(PAPI 7.0.1.0 COMPONENTS sde)
endif()
