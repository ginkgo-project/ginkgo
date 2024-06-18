# Configuring, Building, and Installing

Ginkgo follows the standard CMake procedure.
A default configuration of Ginkgo can be configured, build, and installed with the following three commands:

```shell
cmake -S <path-to-source-directory> -B <path-to-build-directory> [options]
cmake --build <path-to-build-directory>
cmake --install <path-to-build-directory>
```

Ginkgo requires both **CMake 3.16+** and a **C++17 compiler** to be available.
Each backend might have additional requirements.
A full list of the requirements can be found in [](build-install/system-requirements.md).

## Selecting Backends

The default configuration will try to detect and enable all available backends.
Each backend can also be explicitly enabled via the following CMake options:

- `GINKGO_BUILD_REFERENCE`: enable the reference backend
- `GINKGO_BUILD_OMP`: enable the OpenMP backend
- `GINKGO_BUILD_CUDA`: enable the CUDA backend
- `GINKGO_BUILD_HIP`: enable the HIP backend
- `GINKGO_BUILD_SYCL`: enable the SYCL backend

:::{note}
It is valid to enable multiple backends, even GPU ones, at the same time,
as long as the system supports those backends.
:::


Using these, Ginkgo can be told to explicitly enable the CUDA backend and explicitly disable the OpenMP backend
with the following command:

```shell
cmake -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_HIP=OFF -S <source> -B <build>
```

If any backend is explicitly enabled, the required compilers, libraries, etc. must be available on the system.
In the above example, CMake must be able to find a CUDA capable compiler.
This can be helped by setting the CMake option `-DCMAKE_CUDA_COMPILER=/path/to/compiler`.
The same fix can be applied to the HIP backend.
CMake will return an error, if the requirements for enabling a backend are not satisfied.


:::{attention}
If using the SYCL backend, the `CMAKE_CXX_COMPILER` option must be set to a SYCL compatible compiler.
:::

## Configuration Options

Ginkgo has more options than the backend selection to configure its build.
The full list of options is available in [](build-install/cmake-options.md)


## Linking Ginkgo

After installing, Ginkgo may be linked against by using:

```cmake
find_package(Ginkgo VERSION 1.9.0 REQUIRED)

target_link_libraries(target Ginkgo::ginkgo)
```

If Ginkgo can't be found by CMake, either add the install directory to the CMake option `CMAKE_PREFIX_PATH`,
or passing `-DGinkgo_ROOT=<install-directory>` to the CMake configuration of the consumer library.
The default install directory is system dependent.
To explicitly set this directory either provide a prefix path to the install step:

```shell
cmake --install <build-directory> --prefix <custom-install-directory>
```

Alternatively, the same path might be set during the configuration step:

```shell
cmake -DCMAKE_INSTALL_PREFIX=<custom-install-directory> -S <source-directory> -B <build-directory>
```

## Advanced Topics

Details on more advanced topics can be found here:

- [](build-install/tpl.md)
- [](build-install/hip.md)
- [](build-install/windows.md)

```{toctree}
:hidden:

build-install/system-requirements
build-install/cmake-options
build-install/tpl
build-install/hip
build-install/windows
```
