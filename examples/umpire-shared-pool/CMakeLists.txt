cmake_minimum_required(VERSION 3.16)
project(umpire-shared-pool CXX)

# We only need to find Ginkgo if we build this example stand-alone
if(NOT GINKGO_BUILD_EXAMPLES)
    find_package(Ginkgo 1.12.0 REQUIRED)
endif()

# Ginkgo links Umpire privately, so this example has to find it for itself.
# Inside Ginkgo's own build the target exists already, and a second lookup
# would rerun Umpire's CUDA setup.
if(NOT TARGET umpire::umpire AND NOT TARGET umpire)
    find_package(umpire REQUIRED)
endif()
# v2026.07.0 renamed the export to umpire::umpire and deprecated the bare one.
if(TARGET umpire::umpire)
    set(example_umpire_target umpire::umpire)
else()
    set(example_umpire_target umpire)
endif()

add_executable(umpire-shared-pool umpire-shared-pool.cpp)
target_link_libraries(
    umpire-shared-pool
    Ginkgo::ginkgo
    ${example_umpire_target}
)

# Umpire >= 2026.07 requires C++20, while 2025.09 to 2025.12 build at C++17.
if(NOT GINKGO_UMPIRE_CXX_STANDARD)
    if(umpire_VERSION VERSION_LESS 2026.07)
        set(GINKGO_UMPIRE_CXX_STANDARD cxx_std_17)
    else()
        set(GINKGO_UMPIRE_CXX_STANDARD cxx_std_20)
    endif()
endif()
target_compile_features(
    umpire-shared-pool
    PRIVATE ${GINKGO_UMPIRE_CXX_STANDARD}
)
