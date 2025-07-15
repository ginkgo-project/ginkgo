add_custom_target(
    csl
    COMMAND
        ${Python3_EXECUTABLE}
        ${GINKGO_SOURCE_DIR}/csl/application_base_compile.py ${MATRIX_SIZE}
        ${GRID_SIZE} ${GINKGO_SOURCE_DIR}
    COMMENT "Compiling .csl files"
    VERBATIM
)

add_library(ginkgo_csl $<TARGET_OBJECTS:ginkgo_csl_device> "")
target_sources(ginkgo_csl PRIVATE main.cpp)

ginkgo_compile_features(ginkgo_csl)
target_compile_definitions(
    ginkgo_csl
    PRIVATE M=${MATRIX_SIZE} G=${GRID_SIZE} EXECDIR=\"${GINKGO_SOURCE_DIR}\"
)

target_link_libraries(ginkgo_csl PUBLIC ginkgo_device ${Python3_LIBRARIES})
target_include_directories(
    ginkgo_csl
    PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS}
)
target_link_libraries(ginkgo_csl PRIVATE csl)

ginkgo_default_includes(ginkgo_csl)
ginkgo_install_library(ginkgo_csl)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    # FIXME
    # ginkgo_check_headers(ginkgo_csl "GKO_COMPILING_CSL;GKO_DEVICE_NAMESPACE=csl")
endif()

if(GINKGO_BUILD_TESTS)
    # TODO
    # add_subdirectory(test)
endif()
