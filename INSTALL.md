Installation Instructions                      {#install_ginkgo}
-------------------------------------
### Building 

Use the standard cmake build procedure:

```sh
mkdir build; cd build
cmake -G "Unix Makefiles" [OPTIONS] .. && make
```
Use `cmake --build .` in some systems like MinGW or Microsoft Visual Studio which do not use `make`.

For Microsoft Visual Studio, use `cmake --build . --config <build_type>` to decide the build type. The possible options are `Debug`, `Release`, `RelWithDebInfo` and `MinSizeRel`.

Replace `[OPTIONS]` with desired cmake options for your build.
Ginkgo adds the following additional switches to control what is being built:

*   `-DGINKGO_DEVEL_TOOLS={ON, OFF}` sets up the build system for development
    (requires clang-format, will also download git-cmake-format),
    default is `OFF`.
*   `-DGINKGO_BUILD_TESTS={ON, OFF}` builds Ginkgo's tests
    (will download googletest), default is `ON`.
*   `-DGINKGO_BUILD_BENCHMARKS={ON, OFF}` builds Ginkgo's benchmarks
    (will download gflags and rapidjson), default is `ON`.
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
*   `-DGINKGO_BUILD_HIP={ON, OFF}` builds optimized HIP versions of the kernels
    (requires HIP), default is `ON` if an installation of HIP could be detected,
    `OFF` otherwise.
*   `-DGINKGO_HIP_AMDGPU="gpuarch1;gpuarch2"` the amdgpu_target(s) variable
    passed to hipcc for the `hcc` HIP backend. The default is none (auto).
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
*   `-DGINKGO_WITH_CLANG_TIDY={ON, OFF}` makes Ginkgo call `clang-tidy` to find
    programming issues. The path can be manually controlled with the CMake
    variable `-DGINKGO_CLANG_TIDY_PATH=<path>`. The default is `OFF`.
*   `-DGINKGO_WITH_IWYU={ON, OFF}` makes Ginkgo call `iwyu` to find include
    issues. The path can be manually controlled with the CMake variable
    `-DGINKGO_IWYU_PATH=<path>`. The default is `OFF`.
*   `-DGINKGO_VERBOSE_LEVEL=integer` sets the verbosity of Ginkgo.
    * `0` disables all output in the main libraries,
    * `1` enables a few important messages related to unexpected behavior (default).
*   `-DCMAKE_INSTALL_PREFIX=path` sets the installation path for `make install`.
    The default value is usually something like `/usr/local`.
*   `-DCMAKE_BUILD_TYPE=type` specifies which configuration will be used for
    this build of Ginkgo. The default is `RELEASE`. Supported values are CMake's
    standard build types such as `DEBUG` and `RELEASE` and the Ginkgo specific 
	`COVERAGE`, `ASAN` (AddressSanitizer) and `TSAN` (ThreadSanitizer) types.
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
* `-DGINKGO_WINDOWS_SHARED_LIBRARY_RELPATH=<path>` where <path> is a relative
    path built with `PROJECT_BINARY_DIR`. Users must add the absolute path
    (`PROJECT_BINARY_DIR`/`GINKGO_WINDOWS_SHARED_LIBRARY_RELPATH`) into the
    environment variable PATH when building shared libraries and executable
    program, default is `windows_shared_library`.
* `-DGINKGO_CHECK_PATH={ON, OFF}` checks if the environment variable PATH is valid.
    It is checked only when building shared libraries and executable program,
    default is `ON`.

For example, to build everything (in debug mode), use:

```cmake
cmake  -G "Unix Makefiles" -H. -BDebug -DCMAKE_BUILD_TYPE=Debug -DGINKGO_DEVEL_TOOLS=ON \
    -DGINKGO_BUILD_TESTS=ON -DGINKGO_BUILD_REFERENCE=ON -DGINKGO_BUILD_OMP=ON \
    -DGINKGO_BUILD_CUDA=ON -DGINKGO_BUILD_HIP=ON
cmake --build Debug
```

NOTE: Ginkgo is known to work with the `Unix Makefiles` and `Ninja` based
generators. Other CMake generators are untested.

### Building Ginkgo with HIP support
Ginkgo provides a [HIP](https://github.com/ROCm-Developer-Tools/HIP) backend.
This allows to compile optimized versions of the kernels for either AMD or
NVIDIA GPUs. The CMake configuration step will try to auto-detect the presence
of HIP either at `/opt/rocm/hip` or at the path specified by `HIP_PATH` as a
CMake parameter (`-DHIP_PATH=`) or environment variable (`export HIP_PATH=`),
unless `-DGINKGO_BUILD_HIP=ON/OFF` is set explicitly.

#### Correctly installing HIP toolkits and dependencies for Ginkgo
In general, Ginkgo's HIP backend requires the following packages:
+ HIP,
+ hipBLAS,
+ hipSPARSE,
+ Thrust.

It is necessary to provide some details about the different ways to
procure and install these packages, in particular for NVIDIA systems since
getting a correct, non bloated setup is not straightforward.

For AMD systems, the simplest way is to follow the [instructions provided
here](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md) which
provide package installers for most Linux distributions. Ginkgo also needs the
installation of the [hipBLAS](https://github.com/ROCmSoftwarePlatform/hipBLAS)
and [hipSPARSE](https://github.com/ROCmSoftwarePlatform/hipSPARSE) interfaces.
Optionally if you do not already have a thrust installation, [the ROCm provided
rocThrust package can be
used](https://github.com/ROCmSoftwarePlatform/rocThrust).

For NVIDIA systems, the traditional installation (package `hip_nvcc`), albeit
working properly is currently odd: it depends on all the `hcc` related packages,
although the `nvcc` backend seems to entirely rely on the CUDA suite. [See this
issue for more
details](https://github.com/ROCmSoftwarePlatform/hipBLAS/issues/53). It is
advised in this case to compile everything manually, including using forks of
`hipBLAS` and `hipSPARSE` specifically made to not depend on the `hcc` specific
packages. `Thrust` is often provided by CUDA and this Thrust version should work
with `HIP`. Here is a sample procedure for installing `HIP`, `hipBLAS` and
`hipSPARSE`.


```bash
# HIP
git clone https://github.com/ROCm-Developer-Tools/HIP.git
pushd HIP && mkdir build && pushd build
cmake .. && make install
popd && popd

# hipBLAS
git clone https://github.com/tcojean/hipBLAS.git
pushd hipBLAS && mkdir build && pushd build
cmake .. && make install
popd && popd

# hipSPARSE
git clone https://github.com/tcojean/hipSPARSE.git
pushd hipSPARSE && mkdir build && pushd build
cmake -DBUILD_CUDA=ON .. && make install
popd && popd
```


#### Changing the paths to search for HIP and other packages
All HIP installation paths can be configured through the use of environment
variables or CMake variables. This way of configuring the paths is currently
imposed by the `HIP` tool suite. The variables are the following:
+ CMake `-DHIP_PATH=` or  environment `export HIP_PATH=`: sets the `HIP`
  installation path. The default value is `/opt/rocm/hip`.
+ CMake `-DHIPBLAS_PATH=` or  environment `export HIPBLAS_PATH=`: sets the
  `hipBLAS` installation path. The default value is `/opt/rocm/hipblas`.
+ CMake `-DHIPSPARSE_PATH=` or  environment `export HIPSPARSE_PATH=`: sets the
  `hipSPARSE` installation path. The default value is `/opt/rocm/hipsparse`.
+ CMake `-DHCC_PATH=` or  environment `export HCC_PATH=`: sets the `HCC`
  installation path, for AMD backends. The default value is `/opt/rocm/hcc`.
+ environment `export CUDA_PATH=`: where `hipcc` can find `CUDA` if it is not in
  the default `/usr/local/cuda` path.


#### HIP platform detection of AMD and NVIDIA
By default, Ginkgo uses the output of `/opt/rocm/hip/bin/hipconfig --platform`
to select the backend. The accepted values are either `hcc` (AMD) or `nvcc`
(NVIDIA). When on an AMD or NVIDIA system, this should output the correct
platform by default. When on a system without GPUs, this should output `hcc` by
default. To change this value, export the environment variable `HIP_PLATFORM`
like so:
```bash
export HIP_PLATFORM=nvcc
```

#### Setting platform specific compilation flags
Platform specific compilation flags can be given through the following
CMake variables:
+ `-DGINKGO_HIP_COMPILER_FLAGS=`: compilation flags given to all platforms.
+ `-DGINKGO_HIP_HCC_COMPILER_FLAGS=`: compilation flags given to AMD platforms.
+ `-DGINKGO_HIP_NVCC_COMPILER_FLAGS=`: compilation flags given to NVIDIA platforms.


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
Ginkgo to try to use an external version of a package. For this, Ginkgo provides
two ways to find packages. To rely on the CMake `find_package` command, use the
CMake option `-DGINKGO_USE_EXTERNAL_<package>=ON`. Note that, if the external
packages were not installed to the default location, the CMake option
`-DCMAKE_PREFIX_PATH=<path-list>` needs to be set to the semicolon (`;`)
separated list of install paths of these external packages. For more
Information, see the [CMake documentation for
CMAKE_PREFIX_PATH](https://cmake.org/cmake/help/v3.9/variable/CMAKE_PREFIX_PATH.html)
for details.

To manually configure the paths, Ginkgo relies on the [standard xSDK Installation
policies](https://xsdk.info/policies/) for all packages except `CAS` (as it is
neither a library nor a header, it cannot be expressed through the `TPL`
format):
+ `-DTPL_ENABLE_<package>=ON`
+ `-DTPL_<package>_LIBRARIES=/path/to/libraries.{so|a}`
+ `-DTPL_<package>_INCLUDE_DIRS=/path/to/header/directory`

When applicable (e.g. for `GTest` libraries), a `;` separated list can be given
to the `TPL_<package>_{LIBRARIES|INCLUDE_DIRS}` variables.

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

