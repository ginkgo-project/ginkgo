configure_file("${Ginkgo_SOURCE_DIR}/matrices/config.hpp.in"
               "${Ginkgo_BINARY_DIR}/matrices/config.hpp" @ONLY)

configure_file("test/ani1.mtx" "${Ginkgo_BINARY_DIR}/matrices/test/ani1.mtx")
