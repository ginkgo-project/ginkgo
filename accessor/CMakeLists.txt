add_library(ginkgo_accessor INTERFACE)
add_library(Ginkgo::accessor ALIAS ginkgo_accessor)

target_compile_features(ginkgo_accessor INTERFACE cxx_std_17)

target_include_directories(
    ginkgo_accessor
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# TODO TAE: (accessor-export): move this line out of here once the accessor is its own
# GitHub repo. It belongs to Ginkgo (it references Ginkgo's `Ginkgo` export set,
# which the standalone accessor repo must not know about), so it moves into the
# FetchContent wrapper ginkgo/third_party/accessor/CMakeLists.txt, right after
# the fetched `ginkgo_accessor` target becomes available.
install(TARGETS ginkgo_accessor EXPORT Ginkgo)
