Installation Instructions                      {#install_ginkgo}
-------------------------------------
### Building

Use the standard CMake build procedure:

```sh
mkdir build; cd build
cmake [OPTIONS] .. && cmake --build .
```

For Microsoft Visual Studio, use `cmake --build . --config <build_type>` to decide the build type. The possible options are `Debug`, `Release`, `RelWithDebInfo` and `MinSizeRel`.

Replace `[OPTIONS]` with desired cmake options for your build.
Ginkgo adds the following additional switches to control what is being built:

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
    otherwise.
*   `-DGINKGO_BUILD_HIP={ON, OFF}` builds optimized HIP versions of the kernels
    (requires HIP), default is `ON` if an installation of HIP could be detected,
    `OFF` otherwise.
*   `-DGINKGO_HIP_AMDGPU="gpuarch1;gpuarch2"` the amdgpu_target(s) variable
    passed to hipcc for the `hcc` HIP backend. The default is none (auto).
*   `-DGINKGO_BUILD_HWLOC={ON, OFF}` builds Ginkgo with HWLOC. If system HWLOC
    is not found, Ginkgo will try to build it. Default is `ON` on Linux. Ginkgo
    does not support HWLOC on Windows/MacOS, so the default is `OFF` on Windows/MacOS.
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
*   `-DGINKGO_VERBOSE_LEVEL=integer` sets the verbosity of Ginkgo.
    * `0` disables all output in the main libraries,
    * `1` enables a few important messages related to unexpected behavior (default).
*   `GINKGO_INSTALL_RPATH` allows setting any RPATH information when installing
    the Ginkgo libraries. If this is `OFF`, the behavior is the same as if all
    other RPATH flags are set to `OFF` as well. The default is `ON`.
*   `GINKGO_INSTALL_RPATH_ORIGIN` adds $ORIGIN (Linux) or @loader_path (MacOS)
    to the installation RPATH. The default is `ON`.
*   `GINKGO_INSTALL_RPATH_DEPENDENCIES` adds the dependencies to the
    installation RPATH. The default is `OFF`.
*   `-DCMAKE_INSTALL_PREFIX=path` sets the installation path for `make install`.
    The default value is usually something like `/usr/local`.
*   `-DCMAKE_BUILD_TYPE=type` specifies which configuration will be used for
    this build of Ginkgo. The default is `RELEASE`. Supported values are CMake's
    standard build types such as `DEBUG` and `RELEASE` and the Ginkgo specific
    `COVERAGE`, `ASAN` (AddressSanitizer), `LSAN` (LeakSanitizer), `TSAN`
    (ThreadSanitizer) and `UBSAN` (undefined behavior sanitizer) types.
*   `-DBUILD_SHARED_LIBS={ON, OFF}` builds ginkgo as shared libraries (`OFF`)
    or as dynamic libraries (`ON`), default is `ON`.
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
    *   `Kepler`, `Maxwell`, `Pascal`, `Volta`, `Turing`, `Ampere`
    *   `CODE`, `CODE(COMPUTE)`, `(COMPUTE)`

    `Auto` will automatically detect the present CUDA-enabled GPU architectures
    in the system. `Kepler`, `Maxwell`, `Pascal`, `Volta` and `Ampere` will add flags for
    all architectures of that particular NVIDIA GPU generation. `COMPUTE` and
    `CODE` are placeholders that should be replaced with compute and code
    numbers (e.g.  for `compute_70` and `sm_70` `COMPUTE` and `CODE` should be
    replaced with `70`. Default is `Auto`.  For a more detailed explanation of
    this option see the
    [`ARCHITECTURES` specification list](https://github.com/ginkgo-project/CudaArchitectureSelector/blob/master/CudaArchitectureSelector.cmake#L58)
    section in the documentation of the CudaArchitectureSelector CMake module.

Additionally, the following CMake options have effect on the build process:

*  `-DCMAKE_EXPORT_PACKAGE_REGISTRY={ON,OFF}` if set to `ON` the build directory will
   be stored in the current user's CMake package registry.

For example, to build everything (in debug mode), use:

```cmake
cmake .. -BDebug -DCMAKE_BUILD_TYPE=Debug -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_BUILD_TESTS=ON -DGINKGO_BUILD_REFERENCE=ON -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_HIP=ON
cmake --build Debug
```

NOTE: Ginkgo is known to work with the `Unix Makefiles`, `Ninja`, `MinGW Makefiles` and `Visual Studio 16 2019` based
generators. Other CMake generators are untested.

### Building Ginkgo in Windows
Depending on the configuration settings, some manual work might be required:
* Build Ginkgo with Debug mode:
  Some Debug build specific issues can appear depending on the machine and environment:
  When you encounter the error message `ld: error: export ordinal too large`, add the compilation flag `-O1`
  by adding `-DCMAKE_CXX_FLAGS=-O1` to the CMake invocation.
* Build Ginkgo in _MinGW_:\
  If encountering the issue `cc1plus.exe: out of memory allocating 65536 bytes`, please follow the workaround in
  [reference](https://www.intel.com/content/www/us/en/programmable/support/support-resources/knowledge-base/embedded/2016/cc1plus-exe--out-of-memory-allocating-65536-bytes.html),
  or trying to compile ginkgo again might work.

### Building Ginkgo with HIP support
Ginkgo provides a [HIP](https://github.com/ROCm-Developer-Tools/HIP) backend.
This allows to compile optimized versions of the kernels for either AMD or
NVIDIA GPUs. The CMake configuration step will try to auto-detect the presence
of HIP either at `/opt/rocm/hip` or at the path specified by `HIP_PATH` as a
CMake parameter (`-DHIP_PATH=`) or environment variable (`export HIP_PATH=`),
unless `-DGINKGO_BUILD_HIP=ON/OFF` is set explicitly.

#### Changing the paths to search for HIP and other packages
All HIP installation paths can be configured through the use of environment
variables or CMake variables. This way of configuring the paths is currently
imposed by the `HIP` tool suite. The variables are the following:
+ CMake `-DROCM_PATH=` or environment `export ROCM_PATH=`: sets the `ROCM`
  installation path. The default value is `/opt/rocm/`.
+ CMake `-DHIP_CLANG_PATH` or environment `export HIP_CLANG_PATH=`: sets the
  `HIP` compatible `clang` binary path. The default value is
  `${ROCM_PATH}/llvm/bin`.
+ CMake `-DHIP_PATH=` or environment `export HIP_PATH=`: sets the `HIP`
  installation path. The default value is `${ROCM_PATH}/hip`.
+ CMake `-DHIPBLAS_PATH=` or environment `export HIPBLAS_PATH=`: sets the
  `hipBLAS` installation path. The default value is `${ROCM_PATH}/hipblas`.
+ CMake `-DHIPSPARSE_PATH=` or environment `export HIPSPARSE_PATH=`: sets the
  `hipSPARSE` installation path. The default value is `${ROCM_PATH}/hipsparse`.
+ CMake `-DHIPFFT_PATH=` or environment `export HIPFFT_PATH=`: sets the
  `hipFFT` installation path. The default value is `${ROCM_PATH}/hipfft`.
+ CMake `-DROCRAND_PATH=` or environment `export ROCRAND_PATH=`: sets the
  `rocRAND` installation path. The default value is `${ROCM_PATH}/rocrand`.
+ CMake `-DHIPRAND_PATH=` or environment `export HIPRAND_PATH=`: sets the
  `hipRAND` installation path. The default value is `${ROCM_PATH}/hiprand`.
+ environment `export CUDA_PATH=`: where `hipcc` can find `CUDA` if it is not in
  the default `/usr/local/cuda` path.


#### HIP platform detection of AMD and NVIDIA
By default, Ginkgo uses the output of `/opt/rocm/hip/bin/hipconfig --platform`
to select the backend. The accepted values are either `hcc` (`amd` with ROCM >=
4.1) or `nvcc` (`nvidia` with ROCM >= 4.1). When on an AMD or NVIDIA system,
this should output the correct platform by default. When on a system without
GPUs, this should output `hcc` by default. To change this value, export the
environment variable `HIP_PLATFORM` like so:
```bash
export HIP_PLATFORM=nvcc # or nvidia for ROCM >= 4.1
```

#### Setting platform specific compilation flags
Platform specific compilation flags can be given through the following CMake
variables:
+ `-DGINKGO_HIP_COMPILER_FLAGS=`: compilation flags given to all platforms.
+ `-DGINKGO_HIP_NVCC_COMPILER_FLAGS=`: compilation flags given to NVIDIA platforms.
+ `-DGINKGO_HIP_CLANG_COMPILER_FLAGS=`: compilation flags given to AMD clang compiler.


### Third party libraries and packages

Ginkgo relies on third party packages in different cases. These third party
packages can be turned off by disabling the relevant options.

+ GINKGO_BUILD_TESTS=ON: Our tests are implemented with [Google
  Test](https://github.com/google/googletest);
+ GINKGO_BUILD_BENCHMARKS=ON: For argument management we use
  [gflags](https://github.com/gflags/gflags) and for JSON parsing we use
  [nlohmann-json](https://github.com/nlohmann/json);
+ GINKGO_BUILD_HWLOC=ON:
  [hwloc](https://www.open-mpi.org/projects/hwloc) to detect and control cores
  and devices.
+ GINKGO_BUILD_HWLOC=ON and GINKGO_BUILD_TESTS=ON:
  [libnuma](https://www.man7.org/linux/man-pages/man3/numa.3.html) is required
  when testing the functions provided through MachineTopology.
+ GINKGO_BUILD_EXAMPLES=ON:
  [OpenCV](https://opencv.org/) is required for some examples, they are disabled when OpenCV is not available.
+ GINKGO_BUILD_DOC=ON:
  [doxygen](https://www.doxygen.nl/) is required to build the documentation and
  additionally [graphviz](https://graphviz.org/) is required to build the class hierarchy graphs.

Ginkgo attempts to use pre-installed versions of these package if they match
version requirements using `find_package`. Otherwise, the configuration step
will download the files for each of the packages `GTest`, `gflags`,
`nlohmann-json` and `hwloc` and build them internally.

Note that, if the external packages were not installed to the default location,
the CMake option `-DCMAKE_PREFIX_PATH=<path-list>` needs to be set to the
semicolon (`;`) separated list of install paths of these external packages. For
more Information, see the [CMake documentation for
CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.9/variable/CMAKE_PREFIX_PATH.html)
for details.

For convenience, the options `GINKGO_INSTALL_RPATH[_.*]` can be used
to bind the installed Ginkgo shared libraries to the path of its dependencies.

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
An example can be found in the [`test_install`](test/test_install/CMakeLists.txt).
