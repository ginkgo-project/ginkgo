message(STATUS "Fetching external highway")
include(FetchContent)
FetchContent_Declare(
    hwy
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.4.0
)
set(HWY_ENABLE_TESTS OFF CACHE INTERNAL "")
set(HWY_ENABLE_CONTRIB OFF CACHE INTERNAL "")
set(HWY_ENABLE_EXAMPLES OFF CACHE INTERNAL "")
set(HWY_ENABLE_INSTALL OFF CACHE INTERNAL "")
set(HWY_FORCE_STATIC_LIBS ON CACHE INTERNAL "")
FetchContent_MakeAvailable(hwy)

# ginkgo_omp links hwy PRIVATE, so a STATIC ginkgo_omp records $<LINK_ONLY:hwy>
# in its exported interface and install(EXPORT Ginkgo) fails on a target that is
# in no export set. Exporting the vendored hwy as Ginkgo::hwy resolves it, and
# cannot collide with a consumer's own unnamespaced find_package(HWY).
include(GNUInstallDirs)
install(
    TARGETS hwy
    EXPORT Ginkgo
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)
