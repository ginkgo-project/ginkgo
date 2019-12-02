cmake_minimum_required(VERSION 3.8)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

if(GINKGO_HAVE_PAPI_SDE)
    find_package(PAPI REQUIRED OPTIONAL_COMPONENTS sde)
endif()

# Needed because of a known issue with CUDA while linking statically.
# For details, see https://gitlab.kitware.com/cmake/cmake/issues/18614
if((NOT GINKGO_BUILD_SHARED_LIBS) AND GINKGO_BUILD_CUDA)
    enable_language(CUDA)
endif()


add_executable(test_install test_install.cpp)
target_compile_features(test_install PUBLIC cxx_std_11)
target_link_libraries(test_install PRIVATE Ginkgo::ginkgo)
