# CMake Option List

The following discusses CMake options that are relevant to building Ginkgo.
They are used by passing them to the configuration step by:

```shell
cmake -S <...> -B <...> -DOPTION1=VAL1 -DOPTION2=VAL2 ...
```

## Ginkgo-Specific Options

All Ginkgo specific options are prefixed with `GINKGO_`.

*   `-DGINKGO_DEVEL_TOOLS={ON, OFF}` sets up the build system for development
    (requires pre-commit, will also download the clang-format pre-commit hook),
    default is `OFF`. The default behavior installs a pre-commit hook, which
    disables git commits.  If it is set to `ON`, a new pre-commit hook for
    formatting will be installed (enabling commits again). In both cases the
    hook may overwrite a user defined pre-commit hook when Ginkgo is used as
    a submodule.
*   `-DGINKGO_MIXED_PRECISION={ON, OFF}` compiles true mixed-precision kernels
    instead of converting data on the fly, default is `OFF`.
    Enabling this flag increases the library size, but improves performance of
    mixed-precision kernels.
*   `-DGINKGO_BUILD_TESTS={ON, OFF}` builds Ginkgo's tests
    (will download googletest), default is `ON`.
*   `-DGINKGO_FAST_TESTS={ON, OFF}` reduces the input sizes for a few slow tests
    to speed them up, default is `OFF`.
*   `-DGINKGO_BUILD_BENCHMARKS={ON, OFF}` builds Ginkgo's benchmarks
    (will download gflags and nlohmann-json), default is `ON`.
*   `-DGINKGO_BUILD_EXAMPLES={ON, OFF}` builds Ginkgo's examples, default is `ON`
*   `-DGINKGO_BUILD_EXTLIB_EXAMPLE={ON, OFF}` builds the interfacing example
    with deal.II, default is `OFF`.
*   `-DGINKGO_BUILD_REFERENCE={ON, OFF}` build reference implementations of the
    kernels, useful for testing, default is `ON`
*   `-DGINKGO_BUILD_OMP={ON, OFF}` builds optimized OpenMP versions of the kernels,
    default is `ON` if the selected C++ compiler supports OpenMP, `OFF` otherwise.
*   `-DGINKGO_BUILD_CUDA={ON, OFF}` builds optimized cuda versions of the kernels
    (requires CUDA), default is `ON` if a CUDA compiler could be detected,
    `OFF` otherwise.
*   `-DGINKGO_BUILD_DPCPP={ON, OFF}` is deprecated. Please use `GINKGO_BUILD_SYCL` instead.
*   `-DGINKGO_BUILD_SYCL={ON, OFF}` builds optimized SYCL versions of the
    kernels (requires `CMAKE_CXX_COMPILER` to be set to the `dpcpp` or `icpx` compiler).
    The default is `ON` if `CMAKE_CXX_COMPILER` is a SYCL compiler, `OFF`
    otherwise. Due to some differences in IEEE 754 floating point numberhandling in the Intel
    SYCL compilers, Ginkgo tests may fail unless compiled with
    `-DCMAKE_CXX_FLAGS=-ffp-model=precise`
*   `-DGINKGO_BUILD_HIP={ON, OFF}` builds optimized HIP versions of the kernels
    (requires HIP), default is `ON` if an installation of HIP could be detected,
    `OFF` otherwise.
*   `-DGINKGO_BUILD_HWLOC={ON, OFF}` builds Ginkgo with HWLOC. Default is `OFF`.
*   `-DGINKGO_BUILD_DOC={ON, OFF}` creates an HTML version of Ginkgo's documentation
    from inline comments in the code. The default is `OFF`.
*   `-DGINKGO_DOC_GENERATE_EXAMPLES={ON, OFF}` generates the documentation of examples
    in Ginkgo. The default is `ON`.
*   `-DGINKGO_DOC_GENERATE_PDF={ON, OFF}` generates a PDF version of Ginkgo's
    documentation from inline comments in the code. The default is `OFF`.
*   `-DGINKGO_DOC_GENERATE_DEV={ON, OFF}` generates the developer version of
    Ginkgo's documentation. The default is `OFF`.
*   `-DGINKGO_WITH_CLANG_TIDY={ON, OFF}` makes Ginkgo call `clang-tidy` to find
    programming issues. The path can be manually controlled with the CMake
    variable `-DGINKGO_CLANG_TIDY_PATH=<path>`. The default is `OFF`.
*   `-DGINKGO_WITH_IWYU={ON, OFF}` makes Ginkgo call `iwyu` to find include
    issues. The path can be manually controlled with the CMake variable
    `-DGINKGO_IWYU_PATH=<path>`. The default is `OFF`.
*   `-DGINKGO_CHECK_CIRCULAR_DEPS={ON, OFF}` enables compile-time checks for
    circular dependencies between different Ginkgo libraries and self-sufficient
    headers. Should only be used for development purposes. The default is `OFF`.
*   `-DGINKGO_VERBOSE_LEVEL={0, 1}` sets the verbosity of Ginkgo.
    * `0` disables all output in the main libraries,
    * `1` enables a few important messages related to unexpected behavior (default).
*   `GINKGO_INSTALL_RPATH` allows setting any RPATH information when installing
    the Ginkgo libraries. If this is `OFF`, the behavior is the same as if all
    other RPATH flags are set to `OFF` as well. The default is `ON`.
*   `GINKGO_INSTALL_RPATH_ORIGIN` adds $ORIGIN (Linux) or @loader_path (MacOS)
    to the installation RPATH. The default is `ON`.
*   `GINKGO_INSTALL_RPATH_DEPENDENCIES` adds the dependencies to the
    installation RPATH. The default is `OFF`.
*   `-DGINKGO_JACOBI_FULL_OPTIMIZATIONS={ON, OFF}` use all the optimizations
    for the CUDA Jacobi algorithm. `OFF` by default. Setting this option to `ON`
    may lead to very slow compile time (>20 minutes) for the
    `jacobi_generate_kernels.cu` file and high memory usage.
*   `-DGINKGO_CUDA_ARCHITECTURES=<list>` where `<list>` is a semicolon (`;`) separated
    list of architectures. Supported values are:

    *   `Auto`
    *   `Kepler`, `Maxwell`, `Pascal`, `Volta`, `Turing`, `Ampere`

    `Auto` will automatically detect the present CUDA-enabled GPU architectures
    in the system. `Kepler`, `Maxwell`, `Pascal`, `Volta` and `Ampere` will add flags for
    all architectures of that particular NVIDIA GPU generation. It is advised to use
    the non-Ginkgo option `CMAKE_CUDA_ARCHITECTURES` as described below instead.


## Important Non-Ginkgo-Specific Options

These are options that as defined by CMake.
Only a selection of the most relevant options is given here, the full list is available
[here](https://cmake.org/cmake/help/latest/manual/cmake-variables.7.html).

*   `-DCMAKE_BUILD_TYPE=type` specifies which configuration will be used for
    this build of Ginkgo. The default is `RELEASE`. Supported values are CMake's
    standard build types such as `DEBUG` and `RELEASE` and the Ginkgo specific
    `COVERAGE`, `ASAN` (AddressSanitizer), `LSAN` (LeakSanitizer), `TSAN`
    (ThreadSanitizer) and `UBSAN` (undefined behavior sanitizer) types. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html#variable:CMAKE_BUILD_TYPE)
*   `-DCMAKE_CXX_COMPILER=path`, `-DCMAKE_CUDA_COMPILER=path`, `-DCMAKE_HIP_COMPILER=path` set
    the compiler for the respective language. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_COMPILER.html)
*   `-DCMAKE_<LANG>_HOST_COMPILER=path` instructs the build system to explicitly
    set <LANG>'s (either CUDA or HIP) host compiler to the path given as argument. By default, <LANG>
    uses its toolchain's host compiler. Setting this option may help if you're
    experiencing linking errors due to ABI incompatibilities. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_LANG_HOST_COMPILER.html#variable:CMAKE_%3CLANG%3E_HOST_COMPILER)
*   `-DCMAKE_CUDA_ARCHITECTURES="gpuarch1;gpuarch2"` the NVIDIA targets to be passed to the compiler.
    If empty, compiler chooses based on the available GPUs. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_CUDA_ARCHITECTURES.html#cmake-cuda-architectures)
*   `-DCMAKE_HIP_ARCHITECTURES="gpuarch1;gpuarch2"` the AMDGPU targets to be passed to the compiler.
    If empty, compiler chooses based on the available GPUs. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_HIP_ARCHITECTURES.html#cmake-hip-architectures)
*   `-DCMAKE_INSTALL_PREFIX=path` sets the installation path for `make install`.
    The default value is usually something like `/usr/local`. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_INSTALL_PREFIX.html#variable:CMAKE_INSTALL_PREFIX)
*   `-DBUILD_SHARED_LIBS={ON, OFF}` builds ginkgo as shared libraries (`OFF`)
    or as dynamic libraries (`ON`), default is `ON`. [link](https://cmake.org/cmake/help/latest/variable/BUILD_SHARED_LIBS.html#variable:BUILD_SHARED_LIBS)
*  `-DCMAKE_EXPORT_PACKAGE_REGISTRY={ON,OFF}` if set to `ON` the build directory will
   be stored in the current user's CMake package registry. [link](https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_PACKAGE_REGISTRY.html#cmake-export-package-registry)
