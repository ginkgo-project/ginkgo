find_package(csl REQUIRED HINTS "$ENV{CSLROOT}" "$ENV{CSL_ROOT}")
# use the parameter from cmake
set(GINKGO_CSL_ROOT "${csl_DIR}" PARENT_SCOPE)

add_library(ginkgo_csl $<TARGET_OBJECTS:ginkgo_csl_device> "")
target_sources(
    ginkgo_dpcpp
    PRIVATE
        base/executor.cpp
        base/timer.cpp
        base/scoped_device_id.cpp
)

ginkgo_compile_features(ginkgo_csl)
target_compile_definitions(
    ginkgo_dpcpp
    PRIVATE
        GKO_COMPILING_CSL
        GKO_DEVICE_NAMESPACE=csl
)

set(GINKGO_CSL_FLAGS ${GINKGO_CSL_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_csl PRIVATE "${GINKGO_CSL_FLAGS}")
target_link_libraries(ginkgo_csl PUBLIC ginkgo_device)
target_link_libraries(ginkgo_csl PRIVATE csl)

ginkgo_default_includes(ginkgo_dpcpp)
ginkgo_install_library(ginkgo_dpcpp)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_csl "GKO_COMPILING_CSL;GKO_DEVICE_NAMESPACE=csl")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
