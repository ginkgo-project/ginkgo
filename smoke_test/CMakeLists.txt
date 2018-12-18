
cmake_minimum_required(VERSION 3.8)

find_package(Ginkgo REQUIRED
            PATHS # The Path where ginkgo was installed
            # Alternatively, use `cmake -DCMAKE_PREFIX_PATH=<ginkgo_install_dir>` to specify the install directory
            )

add_executable(smoke_test smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE Ginkgo::ginkgo)
