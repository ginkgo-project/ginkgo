Installation Instructions                      {#install_ginkgo}
-------------------------------------
### Building 

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" [OPTIONS] .. && make
```
Replace `[OPTIONS]` with desired cmake options for your build.
Ginkgo adds the following additional switches to control what is being built:

*   `-DGINKGO_DEVEL_TOOLS={ON, OFF}` sets up the build system for development
    (requires clang-format, will also download git-cmake-format),
    default is `ON`
*   `-DGINKGO_BUILD_TESTS={ON, OFF}` builds Ginkgo's tests
    (will download googletest), default is `ON`
*   `-DGINKGO_BUILD_BENCHMARKS={ON, OFF}` builds Ginkgo's benchmarks
    (will download gflags and rapidjson), default is `ON`
*   `-DGINKGO_BUILD_EXAMPLES={ON, OFF}` builds Ginkgo's examples, default is `ON`
*   `-DGINKGO_BUILD_EXTLIB_EXAMPLE={ON, OFF}` builds the interfacing example with deal.II, default is `OFF`
*   `-DGINKGO_BUILD_REFERENCE={ON, OFF}` build reference implementations of the
    kernels, useful for testing, default is `ON`
*   `-DGINKGO_BUILD_OMP={ON, OFF}` builds optimized OpenMP versions of the kernels,
    default is `OFF`
*   `-DGINKGO_BUILD_CUDA={ON, OFF}` builds optimized cuda versions of the kernels
    (requires CUDA), default is `OFF`
*   `-DGINKGO_BUILD_DOC={ON, OFF}` creates an HTML version of Ginkgo's documentation
    from inline comments in the code. The default is `OFF`.
*   `-DGINKGO_DOC_GENERATE_EXAMPLES={ON, OFF}` generates the documentation of examples
     in Ginkgo. The default is `ON`.
*   `-DGINKGO_DOC_GENERATE_PDF={ON, OFF}` generates a PDF version of Ginkgo's
    documentation from inline comments in the code. The default is `OFF`.
*   `-DGINKGO_DOC_GENERATE_DEV={ON, OFF}` generates the developer version of
    Ginkgo's documentation. The default is `OFF`.
*   `-DGINKGO_EXPORT_BUILD_DIR={ON, OFF}` adds the Ginkgo build directory to the
    CMake package registry. The default is `OFF`.
*   `-DGINKGO_VERBOSE_LEVEL=integer` sets the verbosity of Ginkgo.
    * `0` disables all output in the main libraries,
    * `1` enables a few important messages related to unexpected behavior (default).
*   `-DCMAKE_INSTALL_PREFIX=path` sets the installation path for `make install`.
    The default value is usually something like `/usr/local`
*   `-DCMAKE_BUILD_TYPE=type` specifies which configuration will be used for
    this build of Ginkgo. The default is `RELEASE`. Supported values are CMake's
    standard build types such as `DEBUG` and `RELEASE` and the Ginkgo specific 
	`COVERAGE`, `ASAN` (AddressSanitizer) and `TSAN` (ThreadSanitizer) types.
*   `-DBUILD_SHARED_LIBS={ON, OFF}` builds ginkgo as shared libraries (`OFF`)
    or as dynamic libraries (`ON`), default is `ON`
*   `-DGINKGO_JACOBI_FULL_OPTIMIZATIONS={ON, OFF}` use all the optimizations
    for the CUDA Jacobi algorithm. `OFF` by default. Setting this option to `ON`
    may lead to very slow compile time (>20 minutes) for the
    `jacobi_generate_kernels.cu` file and high memory usage.
*   `-DCMAKE_CUDA_HOST_COMPILER=path` instructs the build system to explicitly
    set CUDA's host compiler to the path given as argument. By default, CUDA
    uses its toolchain's host compiler. Setting this option may help if you're
    experiencing linking errors due to ABI incompatibilities. This option is
    supported since [CMake
    3.8](https://github.com/Kitware/CMake/commit/489c52ce680df6439f9c1e553cd2925ca8944cb1)
    but [documented starting from
    3.10](https://cmake.org/cmake/help/v3.10/variable/CMAKE_CUDA_HOST_COMPILER.html).
*   `-DGINKGO_CUDA_ARCHITECTURES=<list>` where `<list>` is a semicolon (`;`) separated
    list of architectures. Supported values are:

    *   `Auto`
    *   `Kepler`, `Maxwell`, `Pascal`, `Volta`
    *   `CODE`, `CODE(COMPUTE)`, `(COMPUTE)`

    `Auto` will automatically detect the present CUDA-enabled GPU architectures
    in the system. `Kepler`, `Maxwell`, `Pascal` and `Volta` will add flags for
    all architectures of that particular NVIDIA GPU generation. `COMPUTE` and
    `CODE` are placeholders that should be replaced with compute and code
    numbers (e.g.  for `compute_70` and `sm_70` `COMPUTE` and `CODE` should be
    replaced with `70`. Default is `Auto`.  For a more detailed explanation of
    this option see the
    [`ARCHITECTURES` specification list](https://github.com/ginkgo-project/CudaArchitectureSelector/blob/master/CudaArchitectureSelector.cmake#L58)
    section in the documentation of the CudaArchitectureSelector CMake module.

For example, to build everything (in debug mode), use:

```cmake
cmake  -G "Unix Makefiles" -H. -BDebug -DCMAKE_BUILD_TYPE=Debug -DGINKGO_DEVEL_TOOLS=ON \
              -DGINKGO_BUILD_TESTS=ON -DGINKGO_BUILD_REFERENCE=ON -DGINKGO_BUILD_OMP=ON \
	          -DGINKGO_BUILD_CUDA=ON 
cmake --build Debug
```

NOTE: Ginkgo is known to work with the `Unix Makefiles` and `Ninja` based
generators. Other CMake generators are untested.

### Third party libraries and packages

Ginkgo relies on third party packages in different cases. These third party
packages can be turned off by disabling the relevant options.

+ GINKGO_BUILD_CUDA=ON:
  [CudaArchitectureSelector](https://github.com/ginkgo-project/CudaArchitectureSelector)
  (CAS) is a CMake helper to manage CUDA architecture settings;
+ GINKGO_BUILD_TESTS=ON: Our tests are implemented with [Google
  Test](https://github.com/google/googletest);
+ GINKGO_BUILD_BENCHMARKS=ON: For argument management we use
  [gflags](https://github.com/gflags/gflags) and for JSON parsing we use
  [RapidJSON](https://github.com/Tencent/rapidjson);
+ GINKGO_DEVEL_TOOLS=ON:
  [git-cmake-format](https://github.com/gflegar/git-cmake-format) is our CMake
  helper for code formatting.

By default, Ginkgo uses the internal version of each package. For each of the
packages `GTEST`, `GFLAGS`, `RAPIDJSON` and `CAS`, it is possible to force
Ginkgo to try to use an external version of a package. For this, set the CMake
option `-DGINKGO_USE_EXTERNAL_<package>=ON`.

If the external packages were not installed to the default location, the
CMake option `-DCMAKE_PREFIX_PATH=<path-list>` needs to be set to the semicolon
(`;`) separated list of install paths of these external packages. For more
Information, see the [CMake documentation for CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.9/variable/CMAKE_PREFIX_PATH.html)
for details.

### Installing Ginkgo

To install Ginkgo into the specified folder, execute the following command in
the build folder

```sh
make install
```

If the installation prefix (see `CMAKE_INSTALL_PREFIX`) is not writable for your
user, e.g. when installing Ginkgo system-wide, it might be necessary to prefix
the call with `sudo`.

After the installation, CMake can find ginkgo with `find_package(Ginkgo)`.
An example can be found in the [`test_install`](test_install/CMakeLists.txt).

