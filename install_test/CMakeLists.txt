cmake_minimum_required(VERSION 3.8)

project(InstallTest LANGUAGES CXX)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

add_executable(install_test install_test.cpp)
target_compile_features(install_test PUBLIC cxx_std_11)
target_link_libraries(install_test PRIVATE Ginkgo::ginkgo)
