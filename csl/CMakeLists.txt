add_custom_target(
    csl
    COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_SOURCE_DIR}/csl/application_base_compile.py
        ${CMAKE_SOURCE_DIR}/csl ${PROJECT_BINARY_DIR}/lib
        "PP:${GRID_SIZE},Mt:${MATRIX_SIZE},Nt:${MATRIX_SIZE},Kt:${MATRIX_SIZE}"
    COMMENT "Compiling .csl files"
    VERBATIM
)

file(GLOB CSL_BYTECODE ${PROJECT_BINARY_DIR}/lib/cs_*.tar.gz)
get_filename_component(CSL_FILENAME ${CSL_BYTECODE} NAME)

set(LIBFOLDER "lib64")

install(
    FILES ${CSL_BYTECODE}
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${LIBFOLDER}
    COMPONENT Runtime
)

add_library(ginkgo_csl $<TARGET_OBJECTS:ginkgo_csl_device> "")
target_sources(
    ginkgo_csl
    PRIVATE
        cerebras_interface.cpp
        base/device.cpp
        base/executor.cpp
        base/scoped_device_id.cpp
        base/timer.cpp
        base/version.cpp
        base/device_hooks.cpp # TODO: remove this later
        matrix/dense_kernels.cpp
)

ginkgo_compile_features(ginkgo_csl)
target_compile_definitions(
    ginkgo_csl
    PRIVATE
        M=${MATRIX_SIZE}
        G=${GRID_SIZE}
        EXECDIR=\"${CMAKE_INSTALL_PREFIX}/${LIBFOLDER}\"
        CSLFILENAME=\"${CSL_FILENAME}\"
        GKO_COMPILING_CSL
        GKO_DEVICE_NAMESPACE=csl
)

target_link_libraries(ginkgo_csl PUBLIC ginkgo_device ${Python3_LIBRARIES})
target_include_directories(
    ginkgo_csl
    PUBLIC ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS}
)
add_dependencies(ginkgo_csl csl)
# target_link_libraries(ginkgo_csl PRIVATE INTERFACE csl)

ginkgo_default_includes(ginkgo_csl)
# ginkgo_install_library(csl)
ginkgo_install_library(ginkgo_csl)

if(GINKGO_CHECK_CIRCULAR_DEPS)
    # FIXME
    ginkgo_check_headers(ginkgo_csl "GKO_COMPILING_CSL;GKO_DEVICE_NAMESPACE=csl")
endif()

if(GINKGO_BUILD_TESTS)
    # TODO
    # add_subdirectory(test)
endif()
