cmake_minimum_required(VERSION 3.8)

project(InstallTest LANGUAGES CXX)

# The `enable_language` is needed when the library is build static with CUDA
#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

add_executable(install_test install_test.cpp)
target_link_libraries(install_test PRIVATE Ginkgo::ginkgo)
