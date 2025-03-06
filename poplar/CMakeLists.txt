find_package(poplar REQUIRED HINTS "$ENV{POPLARROOT}" "$ENV{POPLAR_ROOT}")
# find_package(poplibs REQUIRED HINTS "$ENV{POPLIBSROOT}" "$ENV{POPLIBS_ROOT}" "$ENV{POPLAR_ROOT}")
# use the parameter from cmake
set(GINKGO_POPLAR_ROOT "${poplar_DIR}" PARENT_SCOPE)
# set(GINKGO_POPLIBS_ROOT "${poplibs_DIR}" PARENT_SCOPE)

add_library(ginkgo_poplar $<TARGET_OBJECTS:ginkgo_poplar_device> "")
target_sources(
    ginkgo_dpcpp
    PRIVATE
        base/executor.cpp
        base/timer.cpp
        base/scoped_device_id.cpp
)

ginkgo_compile_features(ginkgo_poplar)
target_compile_definitions(
    ginkgo_dpcpp
    PRIVATE
        GKO_COMPILING_POPLAR
        GKO_DEVICE_NAMESPACE=poplar
)

set(GINKGO_POPLAR_FLAGS ${GINKGO_POPLAR_FLAGS} PARENT_SCOPE)
target_compile_options(ginkgo_poplar PRIVATE "${GINKGO_POPLAR_FLAGS}")
target_link_libraries(ginkgo_poplar PUBLIC ginkgo_device)
target_link_libraries(ginkgo_poplar PRIVATE poplar)

ginkgo_default_includes(ginkgo_dpcpp)
ginkgo_install_library(ginkgo_dpcpp)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    ginkgo_check_headers(ginkgo_poplar "GKO_COMPILING_POPLAR;GKO_DEVICE_NAMESPACE=poplar")
endif()

if(GINKGO_BUILD_TESTS)
    add_subdirectory(test)
endif()
