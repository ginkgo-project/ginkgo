cmake_minimum_required(VERSION 3.8)

project(TestInstall LANGUAGES CXX)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

add_executable(test_install test_install.cpp)
target_compile_features(test_install PUBLIC cxx_std_11)
target_link_libraries(test_install PRIVATE Ginkgo::ginkgo)
