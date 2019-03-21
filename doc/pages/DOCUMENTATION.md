This is the main page for the Ginkgo library user documentation. The repository is hosted on [github](https://github.com/ginkgo-project/ginkgo). Documentation on aspects such as the build system, can be found at the @ref build page. The [example programs](examples.html) can help you get started with using Ginkgo. 

### Modules

The Ginkgo library can be grouped into [modules](modules.html) and these modules form the basic building blocks of Ginkgo. The modules can be summarized as follows:

*   @ref Executor : Where do you want your code to be executed ?
*   @ref LinOp : What kind of operation do you want Ginkgo to perform ?
    * @ref solvers : Solve a linear system for a given matrix.
    * @ref precond : Precondition a system for a solve. 
    * @ref mat_formats : Perform a sparse matrix vector multiplication with a particular matrix format.
*   @ref log : Find out what your code does.
*   @ref stop : When do you want a particular operation to stop ?

@page authors    Authors and contributors

Authors

@page build    Building Ginkgo

### Linux and Mac OS 

For Ginkgo core library:

*   _cmake 3.9+_
*   C++11 compliant compiler, one of:
    *   _gcc 5.3+, 6.3+, 7.3+, 8.1+_
    *   _clang 3.9+_
    *   _Apple LLVM 8.0+_ (__TODO__: verify)

The Ginkgo CUDA module has the following __additional__ requirements:

*   _CUDA 9.0+_
*   Any host compiler restrictions your version of CUDA may impose also apply
    here. For the newest CUDA version, this information can be found in the
    [CUDA installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
    or [CUDA installation guide for Mac Os X](https://docs.nvidia.com/cuda/cuda-installation-guide-mac-os-x/index.html)

In addition, if you want to contribute code to Ginkgo, you will also need the
following:

*   _clang-format 5.0.1+_ (ships as part of _clang_)

### Windows

Windows is currently not supported, but we are working on porting the library
there. If you are interested in helping us with this effort, feel free to
contact one of the developers. (The library itself doesn't use any non-standard
C++ features, so most of the effort here is in modifying the build system.)

__TODO:__ Some restrictions will also apply on the version of C and C++ standard
libraries installed on the system. We need to investigate this further.


Building Ginkgo
---------------

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
*   `-DGINKGO_BUILD_REFERENCE={ON, OFF}` build reference implementations of the
    kernels, useful for testing, default is `OFF`
*   `-DGINKGO_BUILD_OMP={ON, OFF}` builds optimized OpenMP versions of the kernels,
    default is `OFF`
*   `-DGINKGO_BUILD_CUDA={ON, OFF}` builds optimized cuda versions of the kernels
    (requires CUDA), default is `OFF`
*   `-DGINKGO_BUILD_DOC={ON, OFF}` creates an HTML version of Ginkgo's documentation
    from inline comments in the code. The default is `OFF`.
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
packages `GTEST`, `GFLAGS` and `RAPIDJSON` and `CAS`, it is possible to force
Ginkgo to try to use an external version of a package. For this, set the CMake
option `-DGINKGO_USE_EXTERNAL_<package>=ON`.### Installing Ginkgo

### Installation

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

@page test_bench Testing in Ginkgo.

### Running the unit tests
You need to compile ginkgo with `-DGINKGO_BUILD_TESTS=ON` option to be able to run the
tests. 

#### Using make test
After configuring Ginkgo, use the following command inside the build folder to run all tests:

```sh
make test
```

The output should contain several lines of the form:

```
     Start  1: path/to/test
 1/13 Test  #1: path/to/test .............................   Passed    0.01 sec
```

To run only a specific test and see more details results (e.g. if a test failed)
run the following from the build folder:

```sh
./path/to/test
```

where `path/to/test` is the path returned by `make test`.


#### Using CTest 
The tests can also be ran through CTest from the command line, for example when
in a configured build directory:

```sh 
ctest -T start -T build -T test -T submit
```

Will start a new test campaign (usually in `Experimental` mode), build Ginkgo
with the set configuration, run the tests and submit the results to our CDash
dashboard.


Another option is to use Ginkgo's CTest script which is configured to build
Ginkgo with default settings, runs the tests and submits the test to our CDash
dashboard automatically.

To run the script, use the following command:

```sh
ctest -S cmake/CTestScript.cmake
```

The default settings are for our own CI system. Feel free to configure the
script before launching it through variables or by directly changing its values.
A documentation can be found in the script itself.

@page benchmarking Benchmarking with Ginkgo.

### Running the benchmarks

In addition to the unit tests designed to verify correctness, Ginkgo also
includes a benchmark suite for checking its performance on the system. To
compile the benchmarks, the flag `-DGINKGO_BUILD_BENCHMARKS=ON` has to be set during
the `cmake` step. In addition, the [`ssget` command-line
utility](https://github.com/ginkgo-project/ssget) has to be installed on the
system.

The benchmark suite tests Ginkgo's performance using the [SuiteSparse matrix
collection](https://sparse.tamu.edu/) and artificially generated matrices. The
suite sparse collection will be downloaded automatically when the benchmarks are
run. Please note that the entire collection requires roughly 100GB of disk
storage in its compressed format, and roughly 25GB of additional disk space for
intermediate data (such us uncompressing the archive). Additionally, the
benchmark runs usually take a long time (SpMV benchmarks on the complete
collection take roughly 24h using the K20 GPU), and will stress the system.

The benchmark suite is invoked using the `make benchmark` command in the build
directory. The behavior of the suite can be modified using environment
variables. Assuming the `bash` shell is used, these can either be specified via
the `export` command to persist between multiple runs:

```sh
export VARIABLE="value"
...
make benchmark
```

or specified on the fly, on the same line as the `make benchmark` command:

```sh
env VARIABLE="value" ... make benchmark
```

Since `make` sets any variables passed to it as temporary environment variables,
the following shorthand can also be used:

```sh
make benchmark VARIABLE="value" ...
```

A combination of the above approaches is also possible (e.g. it may be useful to
`export` the `SYSTEM_NAME` variable, and specify the others at every benchmark
run).

Supported environment variables are described in the following list:

*   `BENCHMARK={spmv, solver, preconditioner}` - The benchmark set to run.
    Default is `spmv`.
    *   `spmv` - Runs the sparse matrix-vector product benchmarks on the
                 SuiteSparse collection.
    *   `solver` - Runs the solver benchmarks on the SuiteSparse collection.
                The matrix format is determined by running the `spmv` benchmarks
                first, and using the fastest format determined by that
                benchmark. The maximum number of iterations for the iterative
                solvers is set to 10,000 and the requested residual reduction
                factor to 1e-6.
    *   `preconditioner` - Runs the preconditioner benchmarks on artificially
                generated block-diagonal matrices.
*   `DRY_RUN={true, false}` - If set to `true`, prepares the system for the
    benchmark runs (downloads the collections, creates the result structure,
    etc.) and outputs the list of commands that would normally be run, but does
    not run the benchmarks themselves. Default is `false`.
*   `EXECUTOR={reference,cuda,omp}` - The executor used for running the
    benchmarks. Default is `cuda`.
*   `SEGMENTS=<N>` - Splits the benchmark suite into `<N>` segments. This option
    is useful for running the benchmarks on an HPC system with a batch
    scheduler, as it enables partitioning of the benchmark suite and running it
    concurrently on multiple nodes of the system. If specified, `SEGMENT_ID`
    also has to be set. Default is `1`.
*   `SEGMENT_ID=<I>` - used in combination with the `SEGMENTS` variable. `<I>`
    should be an integer between 1 and `<N>`. If specified, only the `<I>`-th
    segment of the benchmark suite will be run. Default is `1`.
*   `SYSTEM_NAME=<name>` - the name of the system where the benchmarks are being
    run. This option only changes the directory where the benchmark results are
    stored. It can be used to avoid overwriting the benchmarks if multiple
    systems share the same filesystem, or when copying the results between
    systems. Default is `unknown`.

Once `make benchmark` completes, the results can be found in
`<Ginkgo build directory>/benchmark/results/${SYSTEM_NAME}/`. The files are
written in the JSON format, and can be analyzed using any of the data
analysis tools that support JSON. Alternatively, they can be uploaded to an
online repository, and analyzed using Ginkgo's free web tool
[Ginkgo Performance Explorer (GPE)](https://ginkgo-project.github.io/gpe/).
(Make sure to change the "Performance data URL" to your repository if using
GPE.)

@page wiki The Ginkgo wiki page.

Ginkgo also has a [wiki page](https://github.com/ginkgo-project/ginkgo/wiki) 

@page known_issues The known issues in Ginkgo.

Please refer to the [Known Issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues). 
