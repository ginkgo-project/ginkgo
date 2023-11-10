# Changelog

This file may not always be up to date in particular for the unreleased
commits. For a comprehensive list, use the following command:
```bash
git log --first-parent
```


## Unreleased

Please visit our wiki [Changelog](https://github.com/ginkgo-project/ginkgo/wiki/Changelog) for unreleased changes.

## Version 1.7.0

The Ginkgo team is proud to announce the new Ginkgo minor release 1.7.0. This release brings new features such as:
- Complete GPU-resident sparse direct solvers feature set and interfaces,
- Improved Cholesky factorization performance,
- A new MC64 reordering,
- Batched iterative solver support with the BiCGSTAB solver with batched Dense and ELL matrix types,
- MPI support for the SYCL backend,
- Improved ParILU(T)/ParIC(T) preconditioner convergence,
and more!

If you face an issue, please first check our [known issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues) and the [open issues list](https://github.com/ginkgo-project/ginkgo/issues) and if you do not find a solution, feel free to [open a new issue](https://github.com/ginkgo-project/ginkgo/issues/new/choose) or ask a question using the [github discussions](https://github.com/ginkgo-project/ginkgo/discussions).

Supported systems and requirements:
+ For all platforms, CMake 3.16+
+ C++14 compliant compiler
+ Linux and macOS
  + GCC: 5.5+
  + clang: 3.9+
  + Intel compiler: 2019+
  + Apple Clang: 14.0 is tested. Earlier versions might also work.
  + NVHPC: 22.7+
  + Cray Compiler: 14.0.1+
  + CUDA module: CMake 3.18+, and CUDA 10.1+ or NVHPC 22.7+
  + HIP module: ROCm 4.5+
  + DPC++ module: Intel oneAPI 2022.1+ with oneMKL and oneDPL. Set the CXX compiler to `dpcpp` or `icpx`.
  + MPI: standard version 3.1+, ideally GPU Aware, for best performance
+ Windows
  + MinGW: GCC 5.5+
  + Microsoft Visual Studio: VS 2019+
  + CUDA module: CUDA 10.1+, Microsoft Visual Studio
  + OpenMP module: MinGW.

### Version support changes

+ CUDA 9.2 is no longer supported and 10.0 is untested [#1382](https://github.com/ginkgo-project/ginkgo/pull/1382)
+ Ginkgo now requires CMake version 3.16 (and 3.18 for CUDA) [#1368](https://github.com/ginkgo-project/ginkgo/pull/1368)

### Interface changes

+ `const` Factory parameters can no longer be modified through `with_*` functions, as this breaks const-correctness [#1336](https://github.com/ginkgo-project/ginkgo/pull/1336) [#1439](https://github.com/ginkgo-project/ginkgo/pull/1439)

### New Deprecations

+ The `device_reset` parameter of CUDA and HIP executors no longer has an effect, and its `allocation_mode` parameters have been deprecated in favor of the `Allocator` interface. [#1315](https://github.com/ginkgo-project/ginkgo/pull/1315)
+ The CMake parameter `GINKGO_BUILD_DPCPP` has been deprecated in favor of `GINKGO_BUILD_SYCL`. [#1350](https://github.com/ginkgo-project/ginkgo/pull/1350)
+ The `gko::reorder::Rcm` interface has been deprecated in favor of `gko::experimental::reorder::Rcm` based on `Permutation`. [#1418](https://github.com/ginkgo-project/ginkgo/pull/1418)
+ The Permutation class' `permute_mask` functionality. [#1415](https://github.com/ginkgo-project/ginkgo/pull/1415)
+ Multiple functions with typos (`set_complex_subpsace()`, range functions such as `conj_operaton` etc). [#1348](https://github.com/ginkgo-project/ginkgo/pull/1348)

### Summary of previous deprecations
+ `gko::lend()` is not necessary anymore.
+ The classes `RelativeResidualNorm` and `AbsoluteResidualNorm` are deprecated in favor of `ResidualNorm`.
+ The class `AmgxPgm` is deprecated in favor of `Pgm`.
+ Default constructors for the CSR `load_balance` and `automatical` strategies
+ The PolymorphicObject's move-semantic `copy_from` variant
+ The templated `SolverBase` class.
+ The class `MachineTopology` is deprecated in favor of `machine_topology`.
+ Logger constructors and create functions with the `executor` parameter.
+ The virtual, protected, Dense functions `compute_norm1_impl`, `add_scaled_impl`, etc.
+ Logger events for solvers and criterion without the additional `implicit_tau_sq` parameter.
+ The global `gko::solver::default_krylov_dim`, use instead `gko::solver::gmres_default_krylov_dim`.

### Added features

+ Adds a batch::BatchLinOp class that forms a base class for batched linear operators such as batched matrix formats, solver and preconditioners [#1379](https://github.com/ginkgo-project/ginkgo/pull/1379)
+ Adds a batch::MultiVector class that enables operations such as dot, norm, scale on batched vectors [#1371](https://github.com/ginkgo-project/ginkgo/pull/1371)
+ Adds a batch::Dense matrix format that stores batched dense matrices and provides gemv operations for these dense matrices. [#1413](https://github.com/ginkgo-project/ginkgo/pull/1413)
+ Adds a batch::Ell matrix format that stores batched Ell matrices and provides spmv operations for these batched Ell matrices. [#1416](https://github.com/ginkgo-project/ginkgo/pull/1416) [#1437](https://github.com/ginkgo-project/ginkgo/pull/1437)
+ Add a batch::Bicgstab solver (class, core, and reference kernels) that enables iterative solution of batched linear systems [#1438](https://github.com/ginkgo-project/ginkgo/pull/1438).
+ Add device kernels (CUDA, HIP, and DPCPP) for batch::Bicgstab solver. [#1443](https://github.com/ginkgo-project/ginkgo/pull/1443).
+ New MC64 reordering algorithm which optimizes the diagonal product or sum of a matrix by permuting the rows, and computes additional scaling factors for equilibriation [#1120](https://github.com/ginkgo-project/ginkgo/pull/1120)
+ New interface for (non-symmetric) permutation and scaled permutation of Dense and Csr matrices [#1415](https://github.com/ginkgo-project/ginkgo/pull/1415)
+ LU and Cholesky Factorizations can now be separated into their factors [#1432](https://github.com/ginkgo-project/ginkgo/pull/1432)
+ New symbolic LU factorization algorithm that is optimized for matrices with an almost-symmetric sparsity pattern [#1445](https://github.com/ginkgo-project/ginkgo/pull/1445)
+ Sorting kernels for SparsityCsr on all backends [#1343](https://github.com/ginkgo-project/ginkgo/pull/1343)
+ Allow passing pre-generated local solver as factory parameter for the distributed Schwarz preconditioner [#1426](https://github.com/ginkgo-project/ginkgo/pull/1426)
+ Add DPCPP kernels for Partition [#1034](https://github.com/ginkgo-project/ginkgo/pull/1034), and CSR's `check_diagonal_entries` and `add_scaled_identity` functionality [#1436](https://github.com/ginkgo-project/ginkgo/pull/1436)
+ Adds a helper function to create a partition based on either local sizes, or local ranges [#1227](https://github.com/ginkgo-project/ginkgo/pull/1227)
+ Add function to compute arithmetic mean of dense and distributed vectors [#1275](https://github.com/ginkgo-project/ginkgo/pull/1275)
+ Adds `icpx` compiler supports [#1350](https://github.com/ginkgo-project/ginkgo/pull/1350)
+ All backends can be built simultaneously [#1333](https://github.com/ginkgo-project/ginkgo/pull/1333)
+ Emits a CMake warning in downstream projects that use different compilers than the installed Ginkgo [#1372](https://github.com/ginkgo-project/ginkgo/pull/1372)
+ Reordering algorithms in sparse_blas benchmark [#1354](https://github.com/ginkgo-project/ginkgo/pull/1354)
+ Benchmarks gained an `-allocator` parameter to specify device allocators [#1385](https://github.com/ginkgo-project/ginkgo/pull/1385)
+ Benchmarks gained an `-input_matrix` parameter that initializes the input JSON based on the filename [#1387](https://github.com/ginkgo-project/ginkgo/pull/1387)
+ Benchmark inputs can now be reordered as a preprocessing step [#1408](https://github.com/ginkgo-project/ginkgo/pull/1408)


### Improvements

+ Significantly improve Cholesky factorization performance [#1366](https://github.com/ginkgo-project/ginkgo/pull/1366)
+ Improve parallel build performance [#1378](https://github.com/ginkgo-project/ginkgo/pull/1378)
+ Allow constrained parallel test execution using CTest resources [#1373](https://github.com/ginkgo-project/ginkgo/pull/1373)
+ Use arithmetic type more inside mixed precision ELL [#1414](https://github.com/ginkgo-project/ginkgo/pull/1414)
+ Most factory parameters of factory type no longer need to be constructed explicitly via `.on(exec)` [#1336](https://github.com/ginkgo-project/ginkgo/pull/1336) [#1439](https://github.com/ginkgo-project/ginkgo/pull/1439)
+ Improve ParILU(T)/ParIC(T) convergence by using more appropriate atomic operations [#1434](https://github.com/ginkgo-project/ginkgo/pull/1434)

### Fixes

+ Fix an over-allocation for OpenMP reductions [#1369](https://github.com/ginkgo-project/ginkgo/pull/1369)
+ Fix DPCPP's common-kernel reduction for empty input sizes [#1362](https://github.com/ginkgo-project/ginkgo/pull/1362)
+ Fix several typos in the API and documentation [#1348](https://github.com/ginkgo-project/ginkgo/pull/1348)
+ Fix inconsistent `Threads` between generations [#1388](https://github.com/ginkgo-project/ginkgo/pull/1388)
+ Fix benchmark median condition [#1398](https://github.com/ginkgo-project/ginkgo/pull/1398)
+ Fix HIP 5.6.0 compilation [#1411](https://github.com/ginkgo-project/ginkgo/pull/1411)
+ Fix missing destruction of rand_generator from cuda/hip [#1417](https://github.com/ginkgo-project/ginkgo/pull/1417)
+ Fix PAPI logger destruction order [#1419](https://github.com/ginkgo-project/ginkgo/pull/1419)
+ Fix TAU logger compilation [#1422](https://github.com/ginkgo-project/ginkgo/pull/1422)
+ Fix relative criterion to not iterate if the residual is already zero [#1079](https://github.com/ginkgo-project/ginkgo/pull/1079)
+ Fix memory_order invocations with C++20 changes [#1402](https://github.com/ginkgo-project/ginkgo/pull/1402)
+ Fix `check_diagonal_entries_exist` report correctly when only missing diagonal value in the last rows. [#1440](https://github.com/ginkgo-project/ginkgo/pull/1440)
+ Fix checking OpenMPI version in cross-compilation settings [#1446](https://github.com/ginkgo-project/ginkgo/pull/1446)
+ Fix false-positive deprecation warnings in Ginkgo, especially for the old Rcm (it doesn't emit deprecation warnings anymore as a result but is still considered deprecated) [#1444](https://github.com/ginkgo-project/ginkgo/pull/1444)

## Version 1.6.0

The Ginkgo team is proud to announce the new Ginkgo minor release 1.6.0. This release brings new features such as:
- Several building blocks for GPU-resident sparse direct solvers like symbolic
  and numerical LU and Cholesky factorization, ...,
- A distributed Schwarz preconditioner,
- New FGMRES and GCR solvers,
- Distributed benchmarks for the SpMV operation, solvers, ...
- Support for non-default streams in the CUDA and HIP backends,
- Mixed precision support for the CSR SpMV,
- A new profiling logger which integrates with NVTX, ROCTX, TAU and VTune to
  provide internal Ginkgo knowledge to most HPC profilers!

and much more.

If you face an issue, please first check our [known issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues) and the [open issues list](https://github.com/ginkgo-project/ginkgo/issues) and if you do not find a solution, feel free to [open a new issue](https://github.com/ginkgo-project/ginkgo/issues/new/choose) or ask a question using the [github discussions](https://github.com/ginkgo-project/ginkgo/discussions).

Supported systems and requirements:
+ For all platforms, CMake 3.13+
+ C++14 compliant compiler
+ Linux and macOS
  + GCC: 5.5+
  + clang: 3.9+
  + Intel compiler: 2018+
  + Apple Clang: 14.0 is tested. Earlier versions might also work.
  + NVHPC: 22.7+
  + Cray Compiler: 14.0.1+
  + CUDA module: CUDA 9.2+ or NVHPC 22.7+
  + HIP module: ROCm 4.5+
  + DPC++ module: Intel OneAPI 2021.3+ with oneMKL and oneDPL. Set the CXX compiler to `dpcpp`.
+ Windows
  + MinGW: GCC 5.5+
  + Microsoft Visual Studio: VS 2019+
  + CUDA module: CUDA 9.2+, Microsoft Visual Studio
  + OpenMP module: MinGW.

### Version Support Changes
+ ROCm 4.0+ -> 4.5+ after [#1303](https://github.com/ginkgo-project/ginkgo/pull/1303)
+ Removed Cygwin pipeline and support [#1283](https://github.com/ginkgo-project/ginkgo/pull/1283)

### Interface Changes
+ Due to internal changes, `ConcreteExecutor::run` will now always throw if the corresponding module for the `ConcreteExecutor` is not build [#1234](https://github.com/ginkgo-project/ginkgo/pull/1234)
+ The constructor of `experimental::distributed::Vector` was changed to only accept local vectors as `std::unique_ptr` [#1284](https://github.com/ginkgo-project/ginkgo/pull/1284)
+ The default parameters for the `solver::MultiGrid` were improved. In particular, the smoother defaults to one iteration of `Ir` with `Jacobi` preconditioner, and the coarse grid solver uses the new direct solver with LU factorization. [#1291](https://github.com/ginkgo-project/ginkgo/pull/1291) [#1327](https://github.com/ginkgo-project/ginkgo/pull/1327)
+ The `iteration_complete` event gained a more expressive overload with additional parameters, the old overloads were deprecated. [#1288](https://github.com/ginkgo-project/ginkgo/pull/1288) [#1327](https://github.com/ginkgo-project/ginkgo/pull/1327)

### Deprecations
+ Deprecated less expressive `iteration_complete` event. Users are advised to now implement the function `void iteration_complete(const LinOp* solver, const LinOp* b, const LinOp* x, const size_type& it, const LinOp* r, const LinOp* tau, const LinOp* implicit_tau_sq, const array<stopping_status>* status, bool stopped)` [#1288](https://github.com/ginkgo-project/ginkgo/pull/1288)

### Added Features
+ A distributed Schwarz preconditioner. [#1248](https://github.com/ginkgo-project/ginkgo/pull/1248)
+ A GCR solver [#1239](https://github.com/ginkgo-project/ginkgo/pull/1239)
+ Flexible Gmres solver [#1244](https://github.com/ginkgo-project/ginkgo/pull/1244)
+ Enable Gmres solver for distributed matrices and vectors [#1201](https://github.com/ginkgo-project/ginkgo/pull/1201)
+ An example that uses Kokkos to assemble the system matrix [#1216](https://github.com/ginkgo-project/ginkgo/pull/1216)
+ A symbolic LU factorization allowing the `gko::experimental::factorization::Lu` and `gko::experimental::solver::Direct` classes to be used for matrices with non-symmetric sparsity pattern [#1210](https://github.com/ginkgo-project/ginkgo/pull/1210)
+ A numerical Cholesky factorization [#1215](https://github.com/ginkgo-project/ginkgo/pull/1215)
+ Symbolic factorizations in host-side operations are now wrapped in a host-side `Operation` to make their execution visible to loggers. This means that profiling loggers and benchmarks are no longer missing a separate entry for their runtime [#1232](https://github.com/ginkgo-project/ginkgo/pull/1232)
+ Symbolic factorization benchmark [#1302](https://github.com/ginkgo-project/ginkgo/pull/1302)
+ The `ProfilerHook` logger allows annotating the Ginkgo execution (apply, operations, ...) for profiling frameworks like NVTX, ROCTX and TAU. [#1055](https://github.com/ginkgo-project/ginkgo/pull/1055)
+ `ProfilerHook::created_(nested_)summary` allows the generation of a lightweight runtime profile over all Ginkgo functions written to a user-defined stream [#1270](https://github.com/ginkgo-project/ginkgo/pull/1270) for both host and device timing functionality [#1313](https://github.com/ginkgo-project/ginkgo/pull/1313)
+ It is now possible to enable host buffers for MPI communications at runtime even if the compile option `GINKGO_FORCE_GPU_AWARE_MPI` is set. [#1228](https://github.com/ginkgo-project/ginkgo/pull/1228)
+ A stencil matrices generator (5-pt, 7-pt, 9-pt, and 27-pt) for benchmarks [#1204](https://github.com/ginkgo-project/ginkgo/pull/1204)
+ Distributed benchmarks (multi-vector blas, SpMV, solver) [#1204](https://github.com/ginkgo-project/ginkgo/pull/1204)
+ Benchmarks for CSR sorting and lookup [#1219](https://github.com/ginkgo-project/ginkgo/pull/1219)
+ A timer for MPI benchmarks that reports the longest time [#1217](https://github.com/ginkgo-project/ginkgo/pull/1217)
+ A `timer_method=min|max|average|median` flag for benchmark timing summary [#1294](https://github.com/ginkgo-project/ginkgo/pull/1294)
+ Support for non-default streams in CUDA and HIP executors [#1236](https://github.com/ginkgo-project/ginkgo/pull/1236)
+ METIS integration for nested dissection reordering [#1296](https://github.com/ginkgo-project/ginkgo/pull/1296)
+ SuiteSparse AMD integration for fillin-reducing reordering [#1328](https://github.com/ginkgo-project/ginkgo/pull/1328)
+ Csr mixed-precision SpMV support [#1319](https://github.com/ginkgo-project/ginkgo/pull/1319)
+ A `with_loggers` function for all `Factory` parameters [#1337](https://github.com/ginkgo-project/ginkgo/pull/1337)

### Improvements
+ Improve naming of kernel operations for loggers [#1277](https://github.com/ginkgo-project/ginkgo/pull/1277)
+ Annotate solver iterations in `ProfilerHook` [#1290](https://github.com/ginkgo-project/ginkgo/pull/1290)
+ Allow using the profiler hooks and inline input strings in benchmarks [#1342](https://github.com/ginkgo-project/ginkgo/pull/1342)
+ Allow passing smart pointers in place of raw pointers to most matrix functions. This means that things like `vec->compute_norm2(x.get())` or `vec->compute_norm2(lend(x))` can be simplified to `vec->compute_norm2(x)` [#1279](https://github.com/ginkgo-project/ginkgo/pull/1279) [#1261](https://github.com/ginkgo-project/ginkgo/pull/1261)
+ Catch overflows in prefix sum operations, which makes Ginkgo's operations much less likely to crash. This also improves the performance of the prefix sum kernel [#1303](https://github.com/ginkgo-project/ginkgo/pull/1303)
+ Make the installed GinkgoConfig.cmake file relocatable and follow more best practices [#1325](https://github.com/ginkgo-project/ginkgo/pull/1325)

### Fixes
+ Fix OpenMPI version check [#1200](https://github.com/ginkgo-project/ginkgo/pull/1200)
+ Fix the mpi cxx type binding by c binding [#1306](https://github.com/ginkgo-project/ginkgo/pull/1306)
+ Fix runtime failures for one-sided MPI wrapper functions observed on some OpenMPI versions [#1249](https://github.com/ginkgo-project/ginkgo/pull/1249)
+ Disable thread pinning with GPU executors due to poor performance [#1230](https://github.com/ginkgo-project/ginkgo/pull/1230)
+ Fix hwloc version detection [#1266](https://github.com/ginkgo-project/ginkgo/pull/1266)
+ Fix PAPI detection in non-implicit include directories [#1268](https://github.com/ginkgo-project/ginkgo/pull/1268)
+ Fix PAPI support for newer PAPI versions: [#1321](https://github.com/ginkgo-project/ginkgo/pull/1321)
+ Fix pkg-config file generation for library paths outside prefix [#1271](https://github.com/ginkgo-project/ginkgo/pull/1271)
+ Fix various build failures with ROCm 5.4, CUDA 12 and OneAPI 6 [#1214](https://github.com/ginkgo-project/ginkgo/pull/1214), [#1235](https://github.com/ginkgo-project/ginkgo/pull/1235), [#1251](https://github.com/ginkgo-project/ginkgo/pull/1251)
+ Fix incorrect read for skew-symmetric MatrixMarket files with explicit diagonal entries [#1272](https://github.com/ginkgo-project/ginkgo/pull/1272)
+ Fix handling of missing diagonal entries in symbolic factorizations [#1263](https://github.com/ginkgo-project/ginkgo/pull/1263)
+ Fix segmentation fault in benchmark matrix construction [#1299](https://github.com/ginkgo-project/ginkgo/pull/1299)
+ Fix the stencil matrix creation for benchmarking [#1305](https://github.com/ginkgo-project/ginkgo/pull/1305)
+ Fix the additional residual check in IR [#1307](https://github.com/ginkgo-project/ginkgo/pull/1307)
+ Fix the cuSPARSE CSR SpMM issue on single strided vector when cuda >= 11.6 [#1322](https://github.com/ginkgo-project/ginkgo/pull/1322) [#1331](https://github.com/ginkgo-project/ginkgo/pull/1331)
+ Fix Isai generation for large sparsity powers [#1327](https://github.com/ginkgo-project/ginkgo/pull/1327)
+ Fix Ginkgo compilation and test with NVHPC >= 22.7 [#1331](https://github.com/ginkgo-project/ginkgo/pull/1331)
+ Fix Ginkgo compilation of 32 bit binaries with MSVC [#1349](https://github.com/ginkgo-project/ginkgo/pull/1349)


## Version 1.5.0

The Ginkgo team is proud to announce the new Ginkgo minor release 1.5.0. This release brings many important new features such as:
- MPI-based multi-node support for all matrix formats and most solvers;
- full DPC++/SYCL support,
- functionality and interface for GPU-resident sparse direct solvers,
- an interface for wrapping solvers with scaling and reordering applied,
- a new algebraic Multigrid solver/preconditioner,
- improved mixed-precision support,
- support for device matrix assembly,

and much more.

If you face an issue, please first check our [known issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues) and the [open issues list](https://github.com/ginkgo-project/ginkgo/issues) and if you do not find a solution, feel free to [open a new issue](https://github.com/ginkgo-project/ginkgo/issues/new/choose) or ask a question using the [github discussions](https://github.com/ginkgo-project/ginkgo/discussions).

Supported systems and requirements:
+ For all platforms, CMake 3.13+
+ C++14 compliant compiler
+ Linux and macOS
  + GCC: 5.5+
  + clang: 3.9+
  + Intel compiler: 2018+
  + Apple LLVM: 8.0+
  + NVHPC: 22.7+
  + Cray Compiler: 14.0.1+
  + CUDA module: CUDA 9.2+ or NVHPC 22.7+
  + HIP module: ROCm 4.0+
  + DPC++ module: Intel OneAPI 2021.3 with oneMKL and oneDPL. Set the CXX compiler to `dpcpp`.
+ Windows
  + MinGW and Cygwin: GCC 5.5+
  + Microsoft Visual Studio: VS 2019
  + CUDA module: CUDA 9.2+, Microsoft Visual Studio
  + OpenMP module: MinGW or Cygwin.


### Algorithm and important feature additions
+ Add MPI-based multi-node for all matrix formats and solvers (except GMRES and IDR). ([#676](https://github.com/ginkgo-project/ginkgo/pull/676), [#908](https://github.com/ginkgo-project/ginkgo/pull/908), [#909](https://github.com/ginkgo-project/ginkgo/pull/909), [#932](https://github.com/ginkgo-project/ginkgo/pull/932), [#951](https://github.com/ginkgo-project/ginkgo/pull/951), [#961](https://github.com/ginkgo-project/ginkgo/pull/961), [#971](https://github.com/ginkgo-project/ginkgo/pull/971), [#976](https://github.com/ginkgo-project/ginkgo/pull/976), [#985](https://github.com/ginkgo-project/ginkgo/pull/985), [#1007](https://github.com/ginkgo-project/ginkgo/pull/1007), [#1030](https://github.com/ginkgo-project/ginkgo/pull/1030), [#1054](https://github.com/ginkgo-project/ginkgo/pull/1054), [#1100](https://github.com/ginkgo-project/ginkgo/pull/1100), [#1148](https://github.com/ginkgo-project/ginkgo/pull/1148))
+ Porting the remaining algorithms (preconditioners like ISAI, Jacobi, Multigrid, ParILU(T) and ParIC(T)) to DPC++/SYCL, update to SYCL 2020, and improve support and performance ([#896](https://github.com/ginkgo-project/ginkgo/pull/896), [#924](https://github.com/ginkgo-project/ginkgo/pull/924), [#928](https://github.com/ginkgo-project/ginkgo/pull/928), [#929](https://github.com/ginkgo-project/ginkgo/pull/929), [#933](https://github.com/ginkgo-project/ginkgo/pull/933), [#943](https://github.com/ginkgo-project/ginkgo/pull/943), [#960](https://github.com/ginkgo-project/ginkgo/pull/960), [#1057](https://github.com/ginkgo-project/ginkgo/pull/1057), [#1110](https://github.com/ginkgo-project/ginkgo/pull/1110),  [#1142](https://github.com/ginkgo-project/ginkgo/pull/1142))
+ Add a Sparse Direct interface supporting GPU-resident numerical LU factorization, symbolic Cholesky factorization, improved triangular solvers, and more ([#957](https://github.com/ginkgo-project/ginkgo/pull/957), [#1058](https://github.com/ginkgo-project/ginkgo/pull/1058), [#1072](https://github.com/ginkgo-project/ginkgo/pull/1072), [#1082](https://github.com/ginkgo-project/ginkgo/pull/1082))
+ Add a ScaleReordered interface that can wrap solvers and automatically apply reorderings and scalings ([#1059](https://github.com/ginkgo-project/ginkgo/pull/1059))
+ Add a Multigrid solver and improve the aggregation based PGM coarsening scheme ([#542](https://github.com/ginkgo-project/ginkgo/pull/542), [#913](https://github.com/ginkgo-project/ginkgo/pull/913), [#980](https://github.com/ginkgo-project/ginkgo/pull/980), [#982](https://github.com/ginkgo-project/ginkgo/pull/982),  [#986](https://github.com/ginkgo-project/ginkgo/pull/986))
+ Add infrastructure for unified, lambda-based, backend agnostic, kernels and utilize it for some simple kernels ([#833](https://github.com/ginkgo-project/ginkgo/pull/833), [#910](https://github.com/ginkgo-project/ginkgo/pull/910), [#926](https://github.com/ginkgo-project/ginkgo/pull/926))
+ Merge different CUDA, HIP, DPC++ and OpenMP tests under a common interface ([#904](https://github.com/ginkgo-project/ginkgo/pull/904), [#973](https://github.com/ginkgo-project/ginkgo/pull/973), [#1044](https://github.com/ginkgo-project/ginkgo/pull/1044), [#1117](https://github.com/ginkgo-project/ginkgo/pull/1117))
+ Add a device_matrix_data type for device-side matrix assembly ([#886](https://github.com/ginkgo-project/ginkgo/pull/886), [#963](https://github.com/ginkgo-project/ginkgo/pull/963), [#965](https://github.com/ginkgo-project/ginkgo/pull/965))
+ Add support for mixed real/complex BLAS operations ([#864](https://github.com/ginkgo-project/ginkgo/pull/864))
+ Add a FFT LinOp for all but DPC++/SYCL ([#701](https://github.com/ginkgo-project/ginkgo/pull/701))
+ Add FBCSR support for NVIDIA and AMD GPUs and CPUs with OpenMP ([#775](https://github.com/ginkgo-project/ginkgo/pull/775))
+ Add CSR scaling ([#848](https://github.com/ginkgo-project/ginkgo/pull/848))
+ Add array::const_view and equivalent to create constant matrices from non-const data ([#890](https://github.com/ginkgo-project/ginkgo/pull/890))
+ Add a RowGatherer LinOp supporting mixed precision to gather dense matrix rows ([#901](https://github.com/ginkgo-project/ginkgo/pull/901))
+ Add mixed precision SparsityCsr SpMV support ([#970](https://github.com/ginkgo-project/ginkgo/pull/970))
+ Allow creating CSR submatrix including from (possibly discontinuous) index sets ([#885](https://github.com/ginkgo-project/ginkgo/pull/885), [#964](https://github.com/ginkgo-project/ginkgo/pull/964))
+ Add a scaled identity addition (M <- aI + bM) feature interface and impls for Csr and Dense ([#942](https://github.com/ginkgo-project/ginkgo/pull/942))


### Deprecations and important changes
+ Deprecate AmgxPgm in favor of the new Pgm name. ([#1149](https://github.com/ginkgo-project/ginkgo/pull/1149)).
+ Deprecate specialized residual norm classes in favor of a common `ResidualNorm` class ([#1101](https://github.com/ginkgo-project/ginkgo/pull/1101))
+ Deprecate CamelCase non-polymorphic types in favor of snake_case versions (like array, machine_topology, uninitialized_array, index_set) ([#1031](https://github.com/ginkgo-project/ginkgo/pull/1031), [#1052](https://github.com/ginkgo-project/ginkgo/pull/1052))
+ Bug fix: restrict gko::share to rvalue references (*possible interface break*) ([#1020](https://github.com/ginkgo-project/ginkgo/pull/1020))
+ Bug fix: when using cuSPARSE's triangular solvers, specifying the factory parameter `num_rhs` is now required when solving for more than one right-hand side, otherwise an exception is thrown ([#1184](https://github.com/ginkgo-project/ginkgo/pull/1184)).
+ Drop official support for old CUDA < 9.2 ([#887](https://github.com/ginkgo-project/ginkgo/pull/887))


### Improved performance additions
+ Reuse tmp storage in reductions in solvers and add a mutable workspace to all solvers ([#1013](https://github.com/ginkgo-project/ginkgo/pull/1013), [#1028](https://github.com/ginkgo-project/ginkgo/pull/1028))
+ Add HIP unsafe atomic option for AMD ([#1091](https://github.com/ginkgo-project/ginkgo/pull/1091))
+ Prefer vendor implementations for Dense dot, conj_dot and norm2 when available ([#967](https://github.com/ginkgo-project/ginkgo/pull/967)).
+ Tuned OpenMP SellP, COO, and ELL SpMV kernels for a small number of RHS ([#809](https://github.com/ginkgo-project/ginkgo/pull/809))


### Fixes
+ Fix various compilation warnings ([#1076](https://github.com/ginkgo-project/ginkgo/pull/1076), [#1183](https://github.com/ginkgo-project/ginkgo/pull/1183), [#1189](https://github.com/ginkgo-project/ginkgo/pull/1189))
+ Fix issues with hwloc-related tests ([#1074](https://github.com/ginkgo-project/ginkgo/pull/1074))
+ Fix include headers for GCC 12 ([#1071](https://github.com/ginkgo-project/ginkgo/pull/1071))
+ Fix for simple-solver-logging example ([#1066](https://github.com/ginkgo-project/ginkgo/pull/1066))
+ Fix for potential memory leak in Logger ([#1056](https://github.com/ginkgo-project/ginkgo/pull/1056))
+ Fix logging of mixin classes ([#1037](https://github.com/ginkgo-project/ginkgo/pull/1037))
+ Improve value semantics for LinOp types, like moved-from state in cross-executor copy/clones ([#753](https://github.com/ginkgo-project/ginkgo/pull/753))
+ Fix some matrix SpMV and conversion corner cases ([#905](https://github.com/ginkgo-project/ginkgo/pull/905), [#978](https://github.com/ginkgo-project/ginkgo/pull/978))
+ Fix uninitialized data ([#958](https://github.com/ginkgo-project/ginkgo/pull/958))
+ Fix CUDA version requirement for cusparseSpSM ([#953](https://github.com/ginkgo-project/ginkgo/pull/953))
+ Fix several issues within bash-script ([#1016](https://github.com/ginkgo-project/ginkgo/pull/1016))
+ Fixes for `NVHPC` compiler support ([#1194](https://github.com/ginkgo-project/ginkgo/pull/1194))


### Other additions
+ Simplify and properly name GMRES kernels ([#861](https://github.com/ginkgo-project/ginkgo/pull/861))
+ Improve pkg-config support for non-CMake libraries ([#923](https://github.com/ginkgo-project/ginkgo/pull/923), [#1109](https://github.com/ginkgo-project/ginkgo/pull/1109))
+ Improve gdb pretty printer ([#987](https://github.com/ginkgo-project/ginkgo/pull/987), [#1114](https://github.com/ginkgo-project/ginkgo/pull/1114))
+ Add a logger highlighting inefficient allocation and copy patterns ([#1035](https://github.com/ginkgo-project/ginkgo/pull/1035))
+ Improved and optimized test random matrix generation ([#954](https://github.com/ginkgo-project/ginkgo/pull/954), [#1032](https://github.com/ginkgo-project/ginkgo/pull/1032))
+ Better CSR strategy defaults ([#969](https://github.com/ginkgo-project/ginkgo/pull/969))
+ Add `move_from` to `PolymorphicObject` ([#997](https://github.com/ginkgo-project/ginkgo/pull/997))
+ Remove unnecessary device_guard usage ([#956](https://github.com/ginkgo-project/ginkgo/pull/956))
+ Improvements to the generic accessor for mixed-precision ([#727](https://github.com/ginkgo-project/ginkgo/pull/727))
+ Add a naive lower triangular solver implementation for CUDA ([#764](https://github.com/ginkgo-project/ginkgo/pull/764))
+ Add support for int64 indices from CUDA 11 onward with SpMV and SpGEMM ([#897](https://github.com/ginkgo-project/ginkgo/pull/897))
+ Add a L1 norm implementation ([#900](https://github.com/ginkgo-project/ginkgo/pull/900))
+ Add reduce_add for arrays ([#831](https://github.com/ginkgo-project/ginkgo/pull/831))
+ Add utility to simplify Dense View creation from an existing Dense vector ([#1136](https://github.com/ginkgo-project/ginkgo/pull/1136)).
+ Add a custom transpose implementation for Fbcsr and Csr transpose for unsupported vendor types ([#1123](https://github.com/ginkgo-project/ginkgo/pull/1123))
+ Make IDR random initialization deterministic ([#1116](https://github.com/ginkgo-project/ginkgo/pull/1116))
+ Move the algorithm choice for triangular solvers from Csr::strategy_type to a factory parameter ([#1088](https://github.com/ginkgo-project/ginkgo/pull/1088))
+ Update CUDA archCoresPerSM ([#1175](https://github.com/ginkgo-project/ginkgo/pull/1116))
+ Add kernels for Csr sparsity pattern lookup ([#994](https://github.com/ginkgo-project/ginkgo/pull/994))
+ Differentiate between structural and numerical zeros in Ell/Sellp ([#1027](https://github.com/ginkgo-project/ginkgo/pull/1027))
+ Add a binary IO format for matrix data ([#984](https://github.com/ginkgo-project/ginkgo/pull/984))
+ Add a tuple zip_iterator implementation ([#966](https://github.com/ginkgo-project/ginkgo/pull/966))
+ Simplify kernel stubs and declarations ([#888](https://github.com/ginkgo-project/ginkgo/pull/888))
+ Simplify GKO_REGISTER_OPERATION with lambdas ([#859](https://github.com/ginkgo-project/ginkgo/pull/859))
+ Simplify copy to device in tests and examples ([#863](https://github.com/ginkgo-project/ginkgo/pull/863))
+ More verbose output to array assertions ([#858](https://github.com/ginkgo-project/ginkgo/pull/858))
+ Allow parallel compilation for Jacobi kernels ([#871](https://github.com/ginkgo-project/ginkgo/pull/871))
+ Change clang-format pointer alignment to left ([#872](https://github.com/ginkgo-project/ginkgo/pull/872))
+ Various improvements and fixes to the benchmarking framework ([#750](https://github.com/ginkgo-project/ginkgo/pull/750), [#759](https://github.com/ginkgo-project/ginkgo/pull/759), [#870](https://github.com/ginkgo-project/ginkgo/pull/870), [#911](https://github.com/ginkgo-project/ginkgo/pull/911), [#1033](https://github.com/ginkgo-project/ginkgo/pull/1033), [#1137](https://github.com/ginkgo-project/ginkgo/pull/1137))
+ Various documentation improvements ([#892](https://github.com/ginkgo-project/ginkgo/pull/892), [#921](https://github.com/ginkgo-project/ginkgo/pull/921), [#950](https://github.com/ginkgo-project/ginkgo/pull/950), [#977](https://github.com/ginkgo-project/ginkgo/pull/977), [#1021](https://github.com/ginkgo-project/ginkgo/pull/1021), [#1068](https://github.com/ginkgo-project/ginkgo/pull/1068), [#1069](https://github.com/ginkgo-project/ginkgo/pull/1069), [#1080](https://github.com/ginkgo-project/ginkgo/pull/1080), [#1081](https://github.com/ginkgo-project/ginkgo/pull/1081), [#1108](https://github.com/ginkgo-project/ginkgo/pull/1108), [#1153](https://github.com/ginkgo-project/ginkgo/pull/1153), [#1154](https://github.com/ginkgo-project/ginkgo/pull/1154))
+ Various CI improvements ([#868](https://github.com/ginkgo-project/ginkgo/pull/868), [#874](https://github.com/ginkgo-project/ginkgo/pull/874), [#884](https://github.com/ginkgo-project/ginkgo/pull/884), [#889](https://github.com/ginkgo-project/ginkgo/pull/889), [#899](https://github.com/ginkgo-project/ginkgo/pull/899), [#903](https://github.com/ginkgo-project/ginkgo/pull/903),  [#922](https://github.com/ginkgo-project/ginkgo/pull/922), [#925](https://github.com/ginkgo-project/ginkgo/pull/925), [#930](https://github.com/ginkgo-project/ginkgo/pull/930), [#936](https://github.com/ginkgo-project/ginkgo/pull/936), [#937](https://github.com/ginkgo-project/ginkgo/pull/937), [#958](https://github.com/ginkgo-project/ginkgo/pull/958), [#882](https://github.com/ginkgo-project/ginkgo/pull/882), [#1011](https://github.com/ginkgo-project/ginkgo/pull/1011), [#1015](https://github.com/ginkgo-project/ginkgo/pull/1015), [#989](https://github.com/ginkgo-project/ginkgo/pull/989), [#1039](https://github.com/ginkgo-project/ginkgo/pull/1039), [#1042](https://github.com/ginkgo-project/ginkgo/pull/1042), [#1067](https://github.com/ginkgo-project/ginkgo/pull/1067), [#1073](https://github.com/ginkgo-project/ginkgo/pull/1073), [#1075](https://github.com/ginkgo-project/ginkgo/pull/1075), [#1083](https://github.com/ginkgo-project/ginkgo/pull/1083), [#1084](https://github.com/ginkgo-project/ginkgo/pull/1084), [#1085](https://github.com/ginkgo-project/ginkgo/pull/1085), [#1139](https://github.com/ginkgo-project/ginkgo/pull/1139), [#1178](https://github.com/ginkgo-project/ginkgo/pull/1178), [#1187](https://github.com/ginkgo-project/ginkgo/pull/1187))


## Version 1.4.0

The Ginkgo team is proud to announce the new Ginkgo minor release 1.4.0. This
release brings most of the Ginkgo functionality to the Intel DPC++ ecosystem
which enables Intel-GPU and CPU execution. The only Ginkgo features which have
not been ported yet are some preconditioners.

Ginkgo's mixed-precision support is greatly enhanced thanks to:
1. The new Accessor concept, which allows writing kernels featuring on-the-fly
memory compression, among other features. The accessor can be used as
header-only, see the [accessor BLAS benchmarks repository](https://github.com/ginkgo-project/accessor-BLAS/tree/develop) as a usage example.
2. All LinOps now transparently support mixed-precision execution. By default,
this is done through a temporary copy which may have a performance impact but
already allows mixed-precision research.

Native mixed-precision ELL kernels are implemented which do not see this cost.
The accessor is also leveraged in a new CB-GMRES solver which allows for
performance improvements by compressing the Krylov basis vectors. Many other
features have been added to Ginkgo, such as reordering support, a new IDR
solver, Incomplete Cholesky preconditioner, matrix assembly support (only CPU
for now), machine topology information, and more!

Supported systems and requirements:
+ For all platforms, cmake 3.13+
+ C++14 compliant compiler
+ Linux and MacOS
  + gcc: 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + clang: 3.9+
  + Intel compiler: 2018+
  + Apple LLVM: 8.0+
  + CUDA module: CUDA 9.0+
  + HIP module: ROCm 4.0+
  + DPC++ module: Intel OneAPI 2021.3. Set the CXX compiler to `dpcpp`.
+ Windows
  + MinGW and Cygwin: gcc 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + Microsoft Visual Studio: VS 2019
  + CUDA module: CUDA 9.0+, Microsoft Visual Studio
  + OpenMP module: MinGW or Cygwin.


### Algorithm and important feature additions
+ Add a new DPC++ Executor for SYCL execution and other base utilities
  [#648](https://github.com/ginkgo-project/ginkgo/pull/648), [#661](https://github.com/ginkgo-project/ginkgo/pull/661), [#757](https://github.com/ginkgo-project/ginkgo/pull/757), [#832](https://github.com/ginkgo-project/ginkgo/pull/832)
+ Port matrix formats, solvers and related kernels to DPC++. For some kernels,
  also make use of a shared kernel implementation for all executors (except
  Reference). [#710](https://github.com/ginkgo-project/ginkgo/pull/710), [#799](https://github.com/ginkgo-project/ginkgo/pull/799), [#779](https://github.com/ginkgo-project/ginkgo/pull/779), [#733](https://github.com/ginkgo-project/ginkgo/pull/733), [#844](https://github.com/ginkgo-project/ginkgo/pull/844), [#843](https://github.com/ginkgo-project/ginkgo/pull/843), [#789](https://github.com/ginkgo-project/ginkgo/pull/789), [#845](https://github.com/ginkgo-project/ginkgo/pull/845), [#849](https://github.com/ginkgo-project/ginkgo/pull/849), [#855](https://github.com/ginkgo-project/ginkgo/pull/855), [#856](https://github.com/ginkgo-project/ginkgo/pull/856)
+ Add accessors which allow multi-precision kernels, among other things.
  [#643](https://github.com/ginkgo-project/ginkgo/pull/643), [#708](https://github.com/ginkgo-project/ginkgo/pull/708)
+ Add support for mixed precision operations through apply in all LinOps. [#677](https://github.com/ginkgo-project/ginkgo/pull/677)
+ Add incomplete Cholesky factorizations and preconditioners as well as some
  improvements to ILU. [#672](https://github.com/ginkgo-project/ginkgo/pull/672), [#837](https://github.com/ginkgo-project/ginkgo/pull/837), [#846](https://github.com/ginkgo-project/ginkgo/pull/846)
+ Add an AMGX implementation and kernels on all devices but DPC++.
  [#528](https://github.com/ginkgo-project/ginkgo/pull/528), [#695](https://github.com/ginkgo-project/ginkgo/pull/695), [#860](https://github.com/ginkgo-project/ginkgo/pull/860)
+ Add a new mixed-precision capability solver, Compressed Basis GMRES
  (CB-GMRES). [#693](https://github.com/ginkgo-project/ginkgo/pull/693), [#763](https://github.com/ginkgo-project/ginkgo/pull/763)
+ Add the IDR(s) solver. [#620](https://github.com/ginkgo-project/ginkgo/pull/620)
+ Add a new fixed-size block CSR matrix format (for the Reference executor).
  [#671](https://github.com/ginkgo-project/ginkgo/pull/671), [#730](https://github.com/ginkgo-project/ginkgo/pull/730)
+ Add native mixed-precision support to the ELL format. [#717](https://github.com/ginkgo-project/ginkgo/pull/717), [#780](https://github.com/ginkgo-project/ginkgo/pull/780)
+ Add Reverse Cuthill-McKee reordering [#500](https://github.com/ginkgo-project/ginkgo/pull/500), [#649](https://github.com/ginkgo-project/ginkgo/pull/649)
+ Add matrix assembly support on CPUs. [#644](https://github.com/ginkgo-project/ginkgo/pull/644)
+ Extends ISAI from triangular to general and spd matrices. [#690](https://github.com/ginkgo-project/ginkgo/pull/690)

### Other additions
+ Add possibility to apply real matrices to complex vectors.
  [#655](https://github.com/ginkgo-project/ginkgo/pull/655), [#658](https://github.com/ginkgo-project/ginkgo/pull/658)
+ Add functions to compute the absolute of a matrix format. [#636](https://github.com/ginkgo-project/ginkgo/pull/636)
+ Add symmetric permutation and improve existing permutations.
  [#684](https://github.com/ginkgo-project/ginkgo/pull/684), [#657](https://github.com/ginkgo-project/ginkgo/pull/657), [#663](https://github.com/ginkgo-project/ginkgo/pull/663)
+ Add a MachineTopology class with HWLOC support [#554](https://github.com/ginkgo-project/ginkgo/pull/554), [#697](https://github.com/ginkgo-project/ginkgo/pull/697)
+ Add an implicit residual norm criterion. [#702](https://github.com/ginkgo-project/ginkgo/pull/702), [#818](https://github.com/ginkgo-project/ginkgo/pull/818), [#850](https://github.com/ginkgo-project/ginkgo/pull/850)
+ Row-major accessor is generalized to more than 2 dimensions and a new
  "block column-major" accessor has been added. [#707](https://github.com/ginkgo-project/ginkgo/pull/707)
+ Add an heat equation example. [#698](https://github.com/ginkgo-project/ginkgo/pull/698), [#706](https://github.com/ginkgo-project/ginkgo/pull/706)
+ Add ccache support in CMake and CI. [#725](https://github.com/ginkgo-project/ginkgo/pull/725), [#739](https://github.com/ginkgo-project/ginkgo/pull/739)
+ Allow tuning and benchmarking variables non intrusively. [#692](https://github.com/ginkgo-project/ginkgo/pull/692)
+ Add triangular solver benchmark [#664](https://github.com/ginkgo-project/ginkgo/pull/664)
+ Add benchmarks for BLAS operations [#772](https://github.com/ginkgo-project/ginkgo/pull/772), [#829](https://github.com/ginkgo-project/ginkgo/pull/829)
+ Add support for different precisions and consistent index types in benchmarks.
  [#675](https://github.com/ginkgo-project/ginkgo/pull/675), [#828](https://github.com/ginkgo-project/ginkgo/pull/828)
+ Add a Github bot system to facilitate development and PR management.
  [#667](https://github.com/ginkgo-project/ginkgo/pull/667), [#674](https://github.com/ginkgo-project/ginkgo/pull/674), [#689](https://github.com/ginkgo-project/ginkgo/pull/689), [#853](https://github.com/ginkgo-project/ginkgo/pull/853)
+ Add Intel (DPC++) CI support and enable CI on HPC systems. [#736](https://github.com/ginkgo-project/ginkgo/pull/736), [#751](https://github.com/ginkgo-project/ginkgo/pull/751), [#781](https://github.com/ginkgo-project/ginkgo/pull/781)
+ Add ssh debugging for Github Actions CI. [#749](https://github.com/ginkgo-project/ginkgo/pull/749)
+ Add pipeline segmentation for better CI speed. [#737](https://github.com/ginkgo-project/ginkgo/pull/737)


### Changes
+ Add a Scalar Jacobi specialization and kernels. [#808](https://github.com/ginkgo-project/ginkgo/pull/808), [#834](https://github.com/ginkgo-project/ginkgo/pull/834), [#854](https://github.com/ginkgo-project/ginkgo/pull/854)
+ Add implicit residual log for solvers and benchmarks. [#714](https://github.com/ginkgo-project/ginkgo/pull/714)
+ Change handling of the conjugate in the dense dot product. [#755](https://github.com/ginkgo-project/ginkgo/pull/755)
+ Improved Dense stride handling. [#774](https://github.com/ginkgo-project/ginkgo/pull/774)
+ Multiple improvements to the OpenMP kernels performance, including COO,
an exclusive prefix sum, and more. [#703](https://github.com/ginkgo-project/ginkgo/pull/703), [#765](https://github.com/ginkgo-project/ginkgo/pull/765), [#740](https://github.com/ginkgo-project/ginkgo/pull/740)
+ Allow specialization of submatrix and other dense creation functions in solvers. [#718](https://github.com/ginkgo-project/ginkgo/pull/718)
+ Improved Identity constructor and treatment of rectangular matrices. [#646](https://github.com/ginkgo-project/ginkgo/pull/646)
+ Allow CUDA/HIP executors to select allocation mode. [#758](https://github.com/ginkgo-project/ginkgo/pull/758)
+ Check if executors share the same memory. [#670](https://github.com/ginkgo-project/ginkgo/pull/670)
+ Improve test install and smoke testing support. [#721](https://github.com/ginkgo-project/ginkgo/pull/721)
+ Update the JOSS paper citation and add publications in the documentation.
  [#629](https://github.com/ginkgo-project/ginkgo/pull/629), [#724](https://github.com/ginkgo-project/ginkgo/pull/724)
+ Improve the version output. [#806](https://github.com/ginkgo-project/ginkgo/pull/806)
+ Add some utilities for dim and span. [#821](https://github.com/ginkgo-project/ginkgo/pull/821)
+ Improved solver and preconditioner benchmarks. [#660](https://github.com/ginkgo-project/ginkgo/pull/660)
+ Improve benchmark timing and output. [#669](https://github.com/ginkgo-project/ginkgo/pull/669), [#791](https://github.com/ginkgo-project/ginkgo/pull/791), [#801](https://github.com/ginkgo-project/ginkgo/pull/801), [#812](https://github.com/ginkgo-project/ginkgo/pull/812)


### Fixes
+ Sorting fix for the Jacobi preconditioner. [#659](https://github.com/ginkgo-project/ginkgo/pull/659)
+ Also log the first residual norm in CGS [#735](https://github.com/ginkgo-project/ginkgo/pull/735)
+ Fix BiCG and HIP CSR to work with complex matrices. [#651](https://github.com/ginkgo-project/ginkgo/pull/651)
+ Fix Coo SpMV on strided vectors. [#807](https://github.com/ginkgo-project/ginkgo/pull/807)
+ Fix segfault of extract_diagonal, add short-and-fat test. [#769](https://github.com/ginkgo-project/ginkgo/pull/769)
+ Fix device_reset issue by moving counter/mutex to device. [#810](https://github.com/ginkgo-project/ginkgo/pull/810)
+ Fix `EnableLogging` superclass. [#841](https://github.com/ginkgo-project/ginkgo/pull/841)
+ Support ROCm 4.1.x and breaking HIP_PLATFORM changes. [#726](https://github.com/ginkgo-project/ginkgo/pull/726)
+ Decreased test size for a few device tests. [#742](https://github.com/ginkgo-project/ginkgo/pull/742)
+ Fix multiple issues with our CMake HIP and RPATH setup.
  [#712](https://github.com/ginkgo-project/ginkgo/pull/712), [#745](https://github.com/ginkgo-project/ginkgo/pull/745), [#709](https://github.com/ginkgo-project/ginkgo/pull/709)
+ Cleanup our CMake installation step. [#713](https://github.com/ginkgo-project/ginkgo/pull/713)
+ Various simplification and fixes to the Windows CMake setup. [#720](https://github.com/ginkgo-project/ginkgo/pull/720), [#785](https://github.com/ginkgo-project/ginkgo/pull/785)
+ Simplify third-party integration. [#786](https://github.com/ginkgo-project/ginkgo/pull/786)
+ Improve Ginkgo device arch flags management. [#696](https://github.com/ginkgo-project/ginkgo/pull/696)
+ Other fixes and improvements to the CMake setup.
  [#685](https://github.com/ginkgo-project/ginkgo/pull/685), [#792](https://github.com/ginkgo-project/ginkgo/pull/792), [#705](https://github.com/ginkgo-project/ginkgo/pull/705), [#836](https://github.com/ginkgo-project/ginkgo/pull/836)
+ Clarification of dense norm documentation [#784](https://github.com/ginkgo-project/ginkgo/pull/784)
+ Various development tools fixes and improvements [#738](https://github.com/ginkgo-project/ginkgo/pull/738), [#830](https://github.com/ginkgo-project/ginkgo/pull/830), [#840](https://github.com/ginkgo-project/ginkgo/pull/840)
+ Make multiple operators/constructors explicit. [#650](https://github.com/ginkgo-project/ginkgo/pull/650), [#761](https://github.com/ginkgo-project/ginkgo/pull/761)
+ Fix some issues, memory leaks and warnings found by MSVC.
  [#666](https://github.com/ginkgo-project/ginkgo/pull/666), [#731](https://github.com/ginkgo-project/ginkgo/pull/731)
+ Improved solver memory estimates and consistent iteration counts [#691](https://github.com/ginkgo-project/ginkgo/pull/691)
+ Various logger improvements and fixes [#728](https://github.com/ginkgo-project/ginkgo/pull/728), [#743](https://github.com/ginkgo-project/ginkgo/pull/743), [#754](https://github.com/ginkgo-project/ginkgo/pull/754)
+ Fix for ForwardIterator requirements in iterator_factory. [#665](https://github.com/ginkgo-project/ginkgo/pull/665)
+ Various benchmark fixes. [#647](https://github.com/ginkgo-project/ginkgo/pull/647), [#673](https://github.com/ginkgo-project/ginkgo/pull/673), [#722](https://github.com/ginkgo-project/ginkgo/pull/722)
+ Various CI fixes and improvements. [#642](https://github.com/ginkgo-project/ginkgo/pull/642), [#641](https://github.com/ginkgo-project/ginkgo/pull/641), [#795](https://github.com/ginkgo-project/ginkgo/pull/795), [#783](https://github.com/ginkgo-project/ginkgo/pull/783), [#793](https://github.com/ginkgo-project/ginkgo/pull/793), [#852](https://github.com/ginkgo-project/ginkgo/pull/852)


## Version 1.3.0

The Ginkgo team is proud to announce the new minor release of Ginkgo version
1.3.0. This release brings CUDA 11 support, changes the default C++ standard to
be C++14 instead of C++11, adds a new Diagonal matrix format and capacity for
diagonal extraction, significantly improves the CMake configuration output
format, adds the Ginkgo paper which got accepted into the Journal of Open Source
Software (JOSS), and fixes multiple issues.

Supported systems and requirements:
+ For all platforms, cmake 3.9+
+ Linux and MacOS
  + gcc: 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + clang: 3.9+
  + Intel compiler: 2017+
  + Apple LLVM: 8.0+
  + CUDA module: CUDA 9.0+
  + HIP module: ROCm 2.8+
+ Windows
  + MinGW and Cygwin: gcc 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + Microsoft Visual Studio: VS 2017 15.7+
  + CUDA module: CUDA 9.0+, Microsoft Visual Studio
  + OpenMP module: MinGW or Cygwin.


The current known issues can be found in the [known issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues).


### Additions
+ Add paper for Journal of Open Source Software (JOSS). [#479](https://github.com/ginkgo-project/ginkgo/pull/479)
+ Add a DiagonalExtractable interface. [#563](https://github.com/ginkgo-project/ginkgo/pull/563)
+ Add a new diagonal Matrix Format. [#580](https://github.com/ginkgo-project/ginkgo/pull/580)
+ Add Cuda11 support. [#603](https://github.com/ginkgo-project/ginkgo/pull/603)
+ Add information output after CMake configuration. [#610](https://github.com/ginkgo-project/ginkgo/pull/610)
+ Add a new preconditioner export example. [#595](https://github.com/ginkgo-project/ginkgo/pull/595)
+ Add a new cuda-memcheck CI job. [#592](https://github.com/ginkgo-project/ginkgo/pull/592)

### Changes
+ Use unified memory in CUDA debug builds. [#621](https://github.com/ginkgo-project/ginkgo/pull/621)
+ Improve `BENCHMARKING.md` with more detailed info. [#619](https://github.com/ginkgo-project/ginkgo/pull/619)
+ Use C++14 standard instead of C++11. [#611](https://github.com/ginkgo-project/ginkgo/pull/611)
+ Update the Ampere sm information and CudaArchitectureSelector. [#588](https://github.com/ginkgo-project/ginkgo/pull/588)

### Fixes
+ Fix documentation warnings and errors. [#624](https://github.com/ginkgo-project/ginkgo/pull/624)
+ Fix warnings for diagonal matrix format. [#622](https://github.com/ginkgo-project/ginkgo/pull/622)
+ Fix criterion factory parameters in CUDA. [#586](https://github.com/ginkgo-project/ginkgo/pull/586)
+ Fix the norm-type in the examples. [#612](https://github.com/ginkgo-project/ginkgo/pull/612)
+ Fix the WAW race in OpenMP is_sorted_by_column_index. [#617](https://github.com/ginkgo-project/ginkgo/pull/617)
+ Fix the example's exec_map by creating the executor only if requested. [#602](https://github.com/ginkgo-project/ginkgo/pull/602)
+ Fix some CMake warnings. [#614](https://github.com/ginkgo-project/ginkgo/pull/614)
+ Fix Windows building documentation. [#601](https://github.com/ginkgo-project/ginkgo/pull/601)
+ Warn when CXX and CUDA host compiler do not match. [#607](https://github.com/ginkgo-project/ginkgo/pull/607)
+ Fix reduce_add, prefix_sum, and doc-build. [#593](https://github.com/ginkgo-project/ginkgo/pull/593)
+ Fix find_library(cublas) issue on machines installing multiple cuda. [#591](https://github.com/ginkgo-project/ginkgo/pull/591)
+ Fix allocator in sellp read. [#589](https://github.com/ginkgo-project/ginkgo/pull/589)
+ Fix the CAS with HIP and NVIDIA backends. [#585](https://github.com/ginkgo-project/ginkgo/pull/585)

### Deletions
+ Remove unused preconditioner parameter in LowerTrs. [#587](https://github.com/ginkgo-project/ginkgo/pull/587)


## Version 1.2.0

The Ginkgo team is proud to announce the new minor release of Ginkgo version
1.2.0. This release brings full HIP support to Ginkgo, new preconditioners
(ParILUT, ISAI), conversion between double and float for all LinOps, and many
more features and fixes.

Supported systems and requirements:
+ For all platforms, cmake 3.9+
+ Linux and MacOS
  + gcc: 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + clang: 3.9+
  + Intel compiler: 2017+
  + Apple LLVM: 8.0+
  + CUDA module: CUDA 9.0+
  + HIP module: ROCm 2.8+
+ Windows
  + MinGW and Cygwin: gcc 5.3+, 6.3+, 7.3+, all versions after 8.1+
  + Microsoft Visual Studio: VS 2017 15.7+
  + CUDA module: CUDA 9.0+, Microsoft Visual Studio
  + OpenMP module: MinGW or Cygwin.


The current known issues can be found in the [known issues page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues).


### Additions
Here are the main additions to the Ginkgo library. Other thematic additions are listed below.
+ Add full HIP support to Ginkgo [#344](https://github.com/ginkgo-project/ginkgo/pull/344), [#357](https://github.com/ginkgo-project/ginkgo/pull/357), [#384](https://github.com/ginkgo-project/ginkgo/pull/384), [#373](https://github.com/ginkgo-project/ginkgo/pull/373), [#391](https://github.com/ginkgo-project/ginkgo/pull/391), [#396](https://github.com/ginkgo-project/ginkgo/pull/396), [#395](https://github.com/ginkgo-project/ginkgo/pull/395), [#393](https://github.com/ginkgo-project/ginkgo/pull/393), [#404](https://github.com/ginkgo-project/ginkgo/pull/404), [#439](https://github.com/ginkgo-project/ginkgo/pull/439), [#443](https://github.com/ginkgo-project/ginkgo/pull/443), [#567](https://github.com/ginkgo-project/ginkgo/pull/567)
+ Add a new ISAI preconditioner [#489](https://github.com/ginkgo-project/ginkgo/pull/489), [#502](https://github.com/ginkgo-project/ginkgo/pull/502), [#512](https://github.com/ginkgo-project/ginkgo/pull/512), [#508](https://github.com/ginkgo-project/ginkgo/pull/508), [#520](https://github.com/ginkgo-project/ginkgo/pull/520)
+ Add support for ParILUT and ParICT factorization with ILU preconditioners [#400](https://github.com/ginkgo-project/ginkgo/pull/400)
+ Add a new BiCG solver [#438](https://github.com/ginkgo-project/ginkgo/pull/438)
+ Add a new permutation matrix format [#352](https://github.com/ginkgo-project/ginkgo/pull/352), [#469](https://github.com/ginkgo-project/ginkgo/pull/469)
+ Add CSR SpGEMM support [#386](https://github.com/ginkgo-project/ginkgo/pull/386), [#398](https://github.com/ginkgo-project/ginkgo/pull/398), [#418](https://github.com/ginkgo-project/ginkgo/pull/418), [#457](https://github.com/ginkgo-project/ginkgo/pull/457)
+ Add CSR SpGEAM support [#556](https://github.com/ginkgo-project/ginkgo/pull/556)
+ Make all solvers and preconditioners transposable [#535](https://github.com/ginkgo-project/ginkgo/pull/535)
+ Add CsrBuilder and CooBuilder for intrusive access to matrix arrays [#437](https://github.com/ginkgo-project/ginkgo/pull/437)
+ Add a standard-compliant allocator based on the Executors [#504](https://github.com/ginkgo-project/ginkgo/pull/504)
+ Support conversions for all LinOp between double and float [#521](https://github.com/ginkgo-project/ginkgo/pull/521)
+ Add a new boolean to the CUDA and HIP executors to control DeviceReset (default off) [#557](https://github.com/ginkgo-project/ginkgo/pull/557)
+ Add a relaxation factor to IR to represent Richardson Relaxation [#574](https://github.com/ginkgo-project/ginkgo/pull/574)
+ Add two new stopping criteria, for relative (to `norm(b)`) and absolute residual norm [#577](https://github.com/ginkgo-project/ginkgo/pull/577)

#### Example additions
+ Templatize all examples to simplify changing the precision [#513](https://github.com/ginkgo-project/ginkgo/pull/513)
+ Add a new adaptive precision block-Jacobi example [#507](https://github.com/ginkgo-project/ginkgo/pull/507)
+ Add a new IR example [#522](https://github.com/ginkgo-project/ginkgo/pull/522)
+ Add a new Mixed Precision Iterative Refinement example [#525](https://github.com/ginkgo-project/ginkgo/pull/525)
+ Add a new example on iterative trisolves in ILU preconditioning [#526](https://github.com/ginkgo-project/ginkgo/pull/526), [#536](https://github.com/ginkgo-project/ginkgo/pull/536), [#550](https://github.com/ginkgo-project/ginkgo/pull/550)

#### Compilation and library changes
+ Auto-detect compilation settings based on environment [#435](https://github.com/ginkgo-project/ginkgo/pull/435), [#537](https://github.com/ginkgo-project/ginkgo/pull/537)
+ Add SONAME to shared libraries [#524](https://github.com/ginkgo-project/ginkgo/pull/524)
+ Add clang-cuda support [#543](https://github.com/ginkgo-project/ginkgo/pull/543)

#### Other additions
+ Add sorting, searching and merging kernels for GPUs [#403](https://github.com/ginkgo-project/ginkgo/pull/403), [#428](https://github.com/ginkgo-project/ginkgo/pull/428), [#417](https://github.com/ginkgo-project/ginkgo/pull/417), [#455](https://github.com/ginkgo-project/ginkgo/pull/455)
+ Add `gko::as` support for smart pointers [#493](https://github.com/ginkgo-project/ginkgo/pull/493)
+ Add setters and getters for criterion factories [#527](https://github.com/ginkgo-project/ginkgo/pull/527)
+ Add a new method to check whether a solver uses `x` as an initial guess [#531](https://github.com/ginkgo-project/ginkgo/pull/531)
+ Add contribution guidelines [#549](https://github.com/ginkgo-project/ginkgo/pull/549)

### Fixes
#### Algorithms
+ Improve the classical CSR strategy's performance [#401](https://github.com/ginkgo-project/ginkgo/pull/401)
+ Improve the CSR automatical strategy [#407](https://github.com/ginkgo-project/ginkgo/pull/407), [#559](https://github.com/ginkgo-project/ginkgo/pull/559)
+ Memory, speed improvements to the ELL kernel [#411](https://github.com/ginkgo-project/ginkgo/pull/411)
+ Multiple improvements and fixes to ParILU [#419](https://github.com/ginkgo-project/ginkgo/pull/419), [#427](https://github.com/ginkgo-project/ginkgo/pull/427), [#429](https://github.com/ginkgo-project/ginkgo/pull/429), [#456](https://github.com/ginkgo-project/ginkgo/pull/456), [#544](https://github.com/ginkgo-project/ginkgo/pull/544)
+ Fix multiple issues with GMRES [#481](https://github.com/ginkgo-project/ginkgo/pull/481), [#523](https://github.com/ginkgo-project/ginkgo/pull/523), [#575](https://github.com/ginkgo-project/ginkgo/pull/575)
+ Optimize OpenMP matrix conversions [#505](https://github.com/ginkgo-project/ginkgo/pull/505)
+ Ensure the linearity of the ILU preconditioner [#506](https://github.com/ginkgo-project/ginkgo/pull/506)
+ Fix IR's use of the advanced apply [#522](https://github.com/ginkgo-project/ginkgo/pull/522)
+ Fix empty matrices conversions and add tests [#560](https://github.com/ginkgo-project/ginkgo/pull/560)

#### Other core functionalities
+ Fix complex number support in our math header [#410](https://github.com/ginkgo-project/ginkgo/pull/410)
+ Fix CUDA compatibility of the main ginkgo header [#450](https://github.com/ginkgo-project/ginkgo/pull/450)
+ Fix isfinite issues [#465](https://github.com/ginkgo-project/ginkgo/pull/465)
+ Fix the Array::view memory leak and the array/view copy/move [#485](https://github.com/ginkgo-project/ginkgo/pull/485)
+ Fix typos preventing use of some interface functions [#496](https://github.com/ginkgo-project/ginkgo/pull/496)
+ Fix the `gko::dim` to abide to the C++ standard [#498](https://github.com/ginkgo-project/ginkgo/pull/498)
+ Simplify the executor copy interface [#516](https://github.com/ginkgo-project/ginkgo/pull/516)
+ Optimize intermediate storage for Composition [#540](https://github.com/ginkgo-project/ginkgo/pull/540)
+ Provide an initial guess for relevant Compositions [#561](https://github.com/ginkgo-project/ginkgo/pull/561)
+ Better management of nullptr as criterion [#562](https://github.com/ginkgo-project/ginkgo/pull/562)
+ Fix the norm calculations for complex support [#564](https://github.com/ginkgo-project/ginkgo/pull/564)

#### CUDA and HIP specific
+ Use the return value of the atomic operations in our wrappers [#405](https://github.com/ginkgo-project/ginkgo/pull/405)
+ Improve the portability of warp lane masks [#422](https://github.com/ginkgo-project/ginkgo/pull/422)
+ Extract thread ID computation into a separate function [#464](https://github.com/ginkgo-project/ginkgo/pull/464)
+ Reorder kernel parameters for consistency [#474](https://github.com/ginkgo-project/ginkgo/pull/474)
+ Fix the use of `pragma unroll` in HIP [#492](https://github.com/ginkgo-project/ginkgo/pull/492)

#### Other
+ Fix the Ginkgo CMake installation files [#414](https://github.com/ginkgo-project/ginkgo/pull/414), [#553](https://github.com/ginkgo-project/ginkgo/pull/553)
+ Fix the Windows compilation [#415](https://github.com/ginkgo-project/ginkgo/pull/415)
+ Always use demangled types in error messages [#434](https://github.com/ginkgo-project/ginkgo/pull/434), [#486](https://github.com/ginkgo-project/ginkgo/pull/486)
+ Add CUDA header dependency to appropriate tests [#452](https://github.com/ginkgo-project/ginkgo/pull/452)
+ Fix several sonarqube or compilation warnings [#453](https://github.com/ginkgo-project/ginkgo/pull/453), [#463](https://github.com/ginkgo-project/ginkgo/pull/463), [#532](https://github.com/ginkgo-project/ginkgo/pull/532), [#569](https://github.com/ginkgo-project/ginkgo/pull/569)
+ Add shuffle tests [#460](https://github.com/ginkgo-project/ginkgo/pull/460)
+ Fix MSVC C2398 error [#490](https://github.com/ginkgo-project/ginkgo/pull/490)
+ Fix missing interface tests in test install [#558](https://github.com/ginkgo-project/ginkgo/pull/558)

### Tools and ecosystem
#### Benchmarks
+ Add better norm support in the benchmarks [#377](https://github.com/ginkgo-project/ginkgo/pull/377)
+ Add CUDA 10.1 generic SpMV support in benchmarks [#468](https://github.com/ginkgo-project/ginkgo/pull/468), [#473](https://github.com/ginkgo-project/ginkgo/pull/473)
+ Add sparse library ILU in benchmarks [#487](https://github.com/ginkgo-project/ginkgo/pull/487)
+ Add overhead benchmarking capacities [#501](https://github.com/ginkgo-project/ginkgo/pull/501)
+ Allow benchmarking from a matrix list file [#503](https://github.com/ginkgo-project/ginkgo/pull/503)
+ Fix benchmarking issue with JSON and non-finite numbers [#514](https://github.com/ginkgo-project/ginkgo/pull/514)
+ Fix benchmark logger crashers with OpenMP [#565](https://github.com/ginkgo-project/ginkgo/pull/565)

#### CI related
+ Improvements to the CI setup with HIP compilation [#421](https://github.com/ginkgo-project/ginkgo/pull/421), [#466](https://github.com/ginkgo-project/ginkgo/pull/466)
+ Add MacOSX CI support [#470](https://github.com/ginkgo-project/ginkgo/pull/470), [#488](https://github.com/ginkgo-project/ginkgo/pull/488)
+ Add Windows CI support [#471](https://github.com/ginkgo-project/ginkgo/pull/471), [#488](https://github.com/ginkgo-project/ginkgo/pull/488), [#510](https://github.com/ginkgo-project/ginkgo/pull/510), [#566](https://github.com/ginkgo-project/ginkgo/pull/566)
+ Use sanitizers instead of valgrind [#476](https://github.com/ginkgo-project/ginkgo/pull/476)
+ Add automatic container generation and update facilities [#499](https://github.com/ginkgo-project/ginkgo/pull/499)
+ Fix the CI parallelism settings [#517](https://github.com/ginkgo-project/ginkgo/pull/517), [#538](https://github.com/ginkgo-project/ginkgo/pull/538), [#539](https://github.com/ginkgo-project/ginkgo/pull/539)
+ Make the codecov patch check informational [#519](https://github.com/ginkgo-project/ginkgo/pull/519)
+ Add support for LLVM sanitizers with improved thread sanitizer support [#578](https://github.com/ginkgo-project/ginkgo/pull/578)

#### Test suite
+ Add an assertion for sparsity pattern equality [#416](https://github.com/ginkgo-project/ginkgo/pull/416)
+ Add core and reference multiprecision tests support [#448](https://github.com/ginkgo-project/ginkgo/pull/448)
+ Speed up GPU tests by avoiding device reset [#467](https://github.com/ginkgo-project/ginkgo/pull/467)
+ Change test matrix location string [#494](https://github.com/ginkgo-project/ginkgo/pull/494)

#### Other
+ Add Ginkgo badges from our tools [#413](https://github.com/ginkgo-project/ginkgo/pull/413)
+ Update the `create_new_algorithm.sh` script [#420](https://github.com/ginkgo-project/ginkgo/pull/420)
+ Bump copyright and improve license management [#436](https://github.com/ginkgo-project/ginkgo/pull/436), [#433](https://github.com/ginkgo-project/ginkgo/pull/433)
+ Set clang-format minimum requirement [#441](https://github.com/ginkgo-project/ginkgo/pull/441), [#484](https://github.com/ginkgo-project/ginkgo/pull/484)
+ Update git-cmake-format [#446](https://github.com/ginkgo-project/ginkgo/pull/446), [#484](https://github.com/ginkgo-project/ginkgo/pull/484)
+ Disable the development tools by default [#442](https://github.com/ginkgo-project/ginkgo/pull/442)
+ Add a script for automatic header formatting [#447](https://github.com/ginkgo-project/ginkgo/pull/447)
+ Add GDB pretty printer for `gko::Array` [#509](https://github.com/ginkgo-project/ginkgo/pull/509)
+ Improve compilation speed [#533](https://github.com/ginkgo-project/ginkgo/pull/533)
+ Add editorconfig support [#546](https://github.com/ginkgo-project/ginkgo/pull/546)
+ Add a compile-time check for header self-sufficiency [#552](https://github.com/ginkgo-project/ginkgo/pull/552)


## Version 1.1.1
This version of Ginkgo provides a few fixes in Ginkgo's core routines. The
supported systems and requirements are unchanged from version 1.1.0.

### Fixes
+ Improve Ginkgo's installation and fix the `test_install` step ([#406](https://github.com/ginkgo-project/ginkgo/pull/406)),
+ Fix some documentation issues ([#406](https://github.com/ginkgo-project/ginkgo/pull/406)),
+ Fix multiple code issues reported by sonarqube ([#406](https://github.com/ginkgo-project/ginkgo/pull/406)),
+ Update the git-cmake-format repository ([#399](https://github.com/ginkgo-project/ginkgo/pull/399)),
+ Improve the global update header script ([#390](https://github.com/ginkgo-project/ginkgo/pull/390)),
+ Fix broken bounds checks ([#388](https://github.com/ginkgo-project/ginkgo/pull/388)),
+ Fix CSR strategies and improve performance ([#379](https://github.com/ginkgo-project/ginkgo/pull/379)),
+ Fix a small typo in the stencil examples ([#381](https://github.com/ginkgo-project/ginkgo/pull/381)),
+ Fix ELL error on small matrices ([#375](https://github.com/ginkgo-project/ginkgo/pull/375)),
+ Fix SellP read function ([#374](https://github.com/ginkgo-project/ginkgo/pull/374)),
+ Add factorization support in `create_new_algorithm.sh`  ([#371](https://github.com/ginkgo-project/ginkgo/pull/371))

## Version 1.1.0

The Ginkgo team is proud to announce the new minor release of Ginkgo version
1.1.0. This release brings several performance improvements, adds Windows support,
adds support for factorizations inside Ginkgo and a new ILU preconditioner
based on ParILU algorithm, among other things. For detailed information, check the respective issue.

Supported systems and requirements:
+ For all platforms, cmake 3.9+
+ Linux and MacOS
  + gcc: 5.3+, 6.3+, 7.3+, 8.1+
  + clang: 3.9+
  + Intel compiler: 2017+
  + Apple LLVM: 8.0+
  + CUDA module: CUDA 9.0+
+ Windows
  + MinGW and Cygwin: gcc 5.3+, 6.3+, 7.3+, 8.1+
  + Microsoft Visual Studio: VS 2017 15.7+
  + CUDA module: CUDA 9.0+, Microsoft Visual Studio
  + OpenMP module: MinGW or Cygwin.


The current known issues can be found in the [known issues
page](https://github.com/ginkgo-project/ginkgo/wiki/Known-Issues).


### Additions
+ Upper and lower triangular solvers ([#327](https://github.com/ginkgo-project/ginkgo/issues/327), [#336](https://github.com/ginkgo-project/ginkgo/issues/336), [#341](https://github.com/ginkgo-project/ginkgo/issues/341), [#342](https://github.com/ginkgo-project/ginkgo/issues/342))
+ New factorization support in Ginkgo, and addition of the ParILU
  algorithm ([#305](https://github.com/ginkgo-project/ginkgo/issues/305), [#315](https://github.com/ginkgo-project/ginkgo/issues/315), [#319](https://github.com/ginkgo-project/ginkgo/issues/319), [#324](https://github.com/ginkgo-project/ginkgo/issues/324))
+ New ILU preconditioner ([#348](https://github.com/ginkgo-project/ginkgo/issues/348), [#353](https://github.com/ginkgo-project/ginkgo/issues/353))
+ Windows MinGW and Cygwin support ([#347](https://github.com/ginkgo-project/ginkgo/issues/347))
+ Windows Visual Studio support ([#351](https://github.com/ginkgo-project/ginkgo/issues/351))
+ New example showing how to use ParILU as a preconditioner ([#358](https://github.com/ginkgo-project/ginkgo/issues/358))
+ New example on using loggers for debugging ([#360](https://github.com/ginkgo-project/ginkgo/issues/360))
+ Add two new 9pt and 27pt stencil examples ([#300](https://github.com/ginkgo-project/ginkgo/issues/300), [#306](https://github.com/ginkgo-project/ginkgo/issues/306))
+ Allow benchmarking CuSPARSE spmv formats through Ginkgo's benchmarks ([#303](https://github.com/ginkgo-project/ginkgo/issues/303))
+ New benchmark for sparse matrix format conversions ([#312](https://github.com/ginkgo-project/ginkgo/issues/312)[#317](https://github.com/ginkgo-project/ginkgo/issues/317))
+ Add conversions between CSR and Hybrid formats ([#302](https://github.com/ginkgo-project/ginkgo/issues/302), [#310](https://github.com/ginkgo-project/ginkgo/issues/310))
+ Support for sorting rows in the CSR format by column indices ([#322](https://github.com/ginkgo-project/ginkgo/issues/322))
+ Addition of a CUDA COO SpMM kernel for improved performance ([#345](https://github.com/ginkgo-project/ginkgo/issues/345))
+ Addition of a LinOp to handle perturbations of the form (identity + scalar *
  basis * projector) ([#334](https://github.com/ginkgo-project/ginkgo/issues/334))
+ New sparsity matrix representation format with Reference and OpenMP
  kernels ([#349](https://github.com/ginkgo-project/ginkgo/issues/349), [#350](https://github.com/ginkgo-project/ginkgo/issues/350))

### Fixes
+ Accelerate GMRES solver for CUDA executor ([#363](https://github.com/ginkgo-project/ginkgo/issues/363))
+ Fix BiCGSTAB solver convergence ([#359](https://github.com/ginkgo-project/ginkgo/issues/359))
+ Fix CGS logging by reporting the residual for every sub iteration ([#328](https://github.com/ginkgo-project/ginkgo/issues/328))
+ Fix CSR,Dense->Sellp conversion's memory access violation ([#295](https://github.com/ginkgo-project/ginkgo/issues/295))
+ Accelerate CSR->Ell,Hybrid conversions on CUDA ([#313](https://github.com/ginkgo-project/ginkgo/issues/313), [#318](https://github.com/ginkgo-project/ginkgo/issues/318))
+ Fixed slowdown of COO SpMV on OpenMP ([#340](https://github.com/ginkgo-project/ginkgo/issues/340))
+ Fix gcc 6.4.0 internal compiler error ([#316](https://github.com/ginkgo-project/ginkgo/issues/316))
+ Fix compilation issue on Apple clang++ 10 ([#322](https://github.com/ginkgo-project/ginkgo/issues/322))
+ Make Ginkgo able to compile on Intel 2017 and above ([#337](https://github.com/ginkgo-project/ginkgo/issues/337))
+ Make the benchmarks spmv/solver use the same matrix formats ([#366](https://github.com/ginkgo-project/ginkgo/issues/366))
+ Fix self-written isfinite function ([#348](https://github.com/ginkgo-project/ginkgo/issues/348))
+ Fix Jacobi issues shown by cuda-memcheck

### Tools and ecosystem improvements
+ Multiple improvements to the CI system and tools ([#296](https://github.com/ginkgo-project/ginkgo/issues/296), [#311](https://github.com/ginkgo-project/ginkgo/issues/311), [#365](https://github.com/ginkgo-project/ginkgo/issues/365))
+ Multiple improvements to the Ginkgo containers ([#328](https://github.com/ginkgo-project/ginkgo/issues/328), [#361](https://github.com/ginkgo-project/ginkgo/issues/361))
+ Add sonarqube analysis to Ginkgo ([#304](https://github.com/ginkgo-project/ginkgo/issues/304), [#308](https://github.com/ginkgo-project/ginkgo/issues/308), [#309](https://github.com/ginkgo-project/ginkgo/issues/309))
+ Add clang-tidy and iwyu support to Ginkgo ([#298](https://github.com/ginkgo-project/ginkgo/issues/298))
+ Improve Ginkgo's support of xSDK M12 policy by adding the `TPL_` arguments
  to CMake ([#300](https://github.com/ginkgo-project/ginkgo/issues/300))
+ Add support for the xSDK R7 policy ([#325](https://github.com/ginkgo-project/ginkgo/issues/325))
+ Fix examples in html documentation ([#367](https://github.com/ginkgo-project/ginkgo/issues/367))

## Version 1.0.0
The Ginkgo team is proud to announce the first release of Ginkgo, the next-generation high-performance on-node sparse linear algebra library. Ginkgo leverages the features of modern C++ to give you a tool for the iterative solution of linear systems that is:

* __Easy to use.__ Interfaces with cryptic naming schemes and dozens of parameters are a thing of the past. Ginkgo was built with good software design in mind, making simple things simple to express.
* __High performance.__ Our optimized CUDA kernels ensure you are reaching the potential of today's GPU-accelerated high-end systems, while Ginkgo's open design allows extension to future hardware architectures.
* __Controllable.__  While Ginkgo can automatically move your data when needed, you remain in control by optionally specifying when the data is moved and what is its ownership scheme.
* __Composable.__ Iterative solution of linear systems is an extremely versatile field, where effective methods are built by mixing and matching various components. Need a GMRES solver preconditioned with a block-Jacobi enhanced BiCGSTAB? Thanks to its novel linear operator abstraction, Ginkgo can do it!
* __Extensible.__ Did not find a component you were looking for? Ginkgo is designed to be easily extended in various ways. You can provide your own loggers, stopping criteria, matrix formats, preconditioners and solvers to Ginkgo and have them integrate as well as the natively supported ones, without the need to modify or recompile the library.

Ease of Use
-----------

Ginkgo uses high level abstractions to develop an efficient and understandable vocabulary for high-performance iterative solution of linear systems. As a result, the solution of a system stored in [matrix market format](https://math.nist.gov/MatrixMarket/formats.html) via a preconditioned Krylov solver on an accelerator is only [20 lines of code away](https://github.com/ginkgo-project/ginkgo/blob/v1.0.0/examples/minimal-cuda-solver/minimal-cuda-solver.cpp):

```c++
#include <ginkgo/ginkgo.hpp>
#include <iostream>

int main()
{
    // Instantiate a CUDA executor
    auto gpu = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    // Read data
    auto A = gko::read<gko::matrix::Csr<>>(std::cin, gpu);
    auto b = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
    auto x = gko::read<gko::matrix::Dense<>>(std::cin, gpu);
    // Create the solver
    auto solver =
        gko::solver::Cg<>::build()
            .with_preconditioner(gko::preconditioner::Jacobi<>::build().on(gpu))
            .with_criteria(
                gko::stop::Iteration::build().with_max_iters(1000u).on(gpu),
                gko::stop::ResidualNormReduction<>::build()
                    .with_reduction_factor(1e-15)
                    .on(gpu))
            .on(gpu);
    // Solve system
    solver->generate(give(A))->apply(lend(b), lend(x));
    // Write result
    write(std::cout, lend(x));
}
```

Notice that Ginkgo is not a tool that generates C++. It _is_ C++. So just [install the library](https://github.com/ginkgo-project/ginkgo/blob/v1.0.0/INSTALL.md) (which is extremely simple due to its CMake-based build system), include the header and start using Ginkgo in your projects.

Already have an existing application and want to use Ginkgo to implement some part of it? Check out our [integration example](https://github.com/ginkgo-project/ginkgo/blob/v1.0.0/examples/three-pt-stencil-solver/three-pt-stencil-solver.cpp#L144) for a demonstration on how Ginkgo can be used with raw data already available in the application. If your data is in one of the formats supported by Ginkgo, it may be possible to use it directly, without creating a Ginkgo-dedicated copy of it.

Designed for HPC
----------------

Ginkgo is designed to quickly adapt to rapid changes in the HPC architecture. Every component in Ginkgo is built around the _executor_ abstraction which is used to describe the execution and memory spaces where the operations are run, and the programming model used to realize the operations. The low-level performance critical kernels are implemented directly using each executor's programming model, while the high-level operations use a unified implementation that calls the low-level kernels. Consequently, the cost of developing new algorithms and extending existing ones to new architectures is kept relatively low, without compromising performance.
Currently, Ginkgo supports CUDA, reference and OpenMP executors.

The CUDA executor features highly-optimized kernels able to efficiently utilize NVIDIA's latest hardware. Several of these kernels appeared in recent scientific publications, including the optimized COO and CSR SpMV, and the block-Jacobi preconditioner with its adaptive precision version.

The reference executor can be used to verify the correctness of the code. It features a straightforward single threaded C++ implementation of the kernels which is easy to understand. As such, it can be used as a baseline for implementing other executors, verifying their correctness, or figuring out if unexpected behavior is the result of a faulty kernel or an error in the user's code.

Ginkgo 1.0.0 also offers initial support for the OpenMP executor. OpenMP kernels are currently implemented as minor modifications of the reference kernels with OpenMP pragmas and are considered experimental. Full OpenMP support with highly-optimized kernels is reserved for a future release.

Memory Management
-----------------

As a result of its executor-based design and high level abstractions, Ginkgo has explicit information about the location of every piece of data it needs and can automatically allocate, free and move the data where it is needed. However, lazily moving data around is often not optimal, and determining when a piece of data should be copied or shared in general cannot be done automatically. For this reason, Ginkgo also gives explicit control of sharing and moving its objects to the user via the dedicated ownership commands: `gko::clone`, `gko::share`, `gko::give` and `gko::lend`. If you are interested in a detailed description of the problems the C++ standard has with these concepts check out [this Ginkgo Wiki page](https://github.com/ginkgo-project/ginkgo/wiki/Library-design#use-of-pointers), and for more details about Ginkgo's solution to the problem and the description of ownership commands take a look at [this issue](https://github.com/ginkgo-project/ginkgo/issues/30).

Components
----------

Instead of providing a single method to solve a linear system, Ginkgo provides a selection of components that can be used to tailor the solver to your specific problem. It is also possible to use each component separately, as part of larger software. The provided components include matrix formats, solvers and preconditioners (commonly referred to as "_linear operators_" in Ginkgo), as well as executors, stopping criteria and loggers.

Matrix formats are used to represent the system matrix and the vectors of the system. The following are the supported matrix formats (see [this Matrix Format wiki page](https://github.com/ginkgo-project/ginkgo/wiki/Matrix-Formats-in-Ginkgo) for more details):

* `gko::matrix::Dense` - the row-major storage dense matrix format;
* `gko::matrix::Csr` - the Compressed Sparse Row (CSR) sparse matrix format;
* `gko::matrix::Coo` - the Coordinate (COO) sparse matrix format;
* `gko::matrix::Ell` - the ELLPACK (ELL) sparse matrix format;
* `gko::matrix::Sellp` - the SELL-P sparse matrix format based on the sliced ELLPACK representation;
* `gko::matrix::Hybrid` - the hybrid matrix format that represents a matrix as a sum of an ELL and COO matrix.

All formats offer support for the `apply` operation that performs a (sparse) matrix-vector product between the matrix and one or multiple vectors. Conversion routines between the formats are also provided. `gko::matrix::Dense` offers an extended interface that includes simple vector operations such as addition, scaling, dot product and norm, which are applied on each column of the matrix separately.
The interface for all operations is designed to allow any type of matrix format as a parameter. However, version 1.0.0 of this library supports only instances of `gko::matrix::Dense` as vector arguments (the matrix arguments do not have any limitations).

Solvers are utilized to solve the system with a given system matrix and right hand side. Currently, you can choose from several high-performance Krylov methods implemented in Ginkgo:

* `gko::solver::Cg` - the Conjugate Gradient method (CG) suitable for symmetric positive definite problems;
* `gko::solver::Fcg` - the flexible variant of Conjugate Gradient (FCG) that supports non-constant preconditioners;
* `gko::solver::Cgs` - the Conjuage Gradient Squared method (CGS) for general problems;
* `gko::solver::Bicgstab` - the BiConjugate Gradient Stabilized method (BiCGSTAB) for general problems;
* `gko::solver::Gmres` - the restarted Generalized Minimal Residual method (GMRES) for general problems.

All solvers work with system matrices stored in any of the matrix formats described above, and any other general _linear operator_, such as combinations and compositions of other operators, or any matrix format you defined specifically for your application.

Preconditioners can be effective at improving the convergence rate of Krylov methods. All solvers listed above are implemented with preconditioning support. This version of Ginkgo has support for one preconditioner type, but stay tuned, as more preconditioners are coming in future releases:

* `gko::preconditioner::Jacobi` - a highly optimized version of the block-Jacobi preconditioner (block-diagonal scaling), optionally enhanced with adaptive precision storage scheme for additional performance gains.

You can use the block-Jacobi preconditioner with system matrices stored in any of the built-in matrix formats and any custom format that has a defined conversion into a CSR matrix.

Any linear operator (matrix, solver, preconditioner) can be combined into complex operators by using the following utilities:

* `gko::Combination` - creates a linear combination **&alpha;<sub>1</sub> A<sub>1</sub> + ... + &alpha;<sub>n</sub> A<sub>n</sub>** of linear operators;
* `gko::Composition` - creates a composition **A<sub>1</sub> ... A<sub>n</sub>** of linear operators.

You can utilize these utilities (together with a solver which represents the inversion operation) to compute complex expressions, such as **x = (3A - B<sup>-1</sup>C)<sup>-1</sup>b**.

As described in the "Designed for HPC" section, you have a choice between 3 different executors:

* `gko::CudaExecutor` - offers a highly optimized GPU implementation tailored for recent HPC systems;
* `gko::ReferenceExecutor` - single-threaded reference implementation for easy development and testing on systems without a GPU;
* `gko::OmpExecutor` - preliminary OpenMP-based implementation for CPUs.

With Ginkgo, you have fine control over the solver iteration process to ensure that you obtain your solution under the time and accuracy constraints of your application. Ginkgo supports the following stopping criteria out of the box:

* `gko::stop::Iteration` - the iteration process is stopped once the specified iteration count is reached;
* `gko::stop::ResidualNormReduction` - the iteration process is stopped once the initial residual norm is reduced by the specified factor;
* `gko::stop::Time` - the iteration process is stopped if the specified time limit is reached.

You can combine multiple criteria to achieve the desired result, and even add your own criteria to the mix.

Ginkgo also allows you to keep track of the events that happen while using the library, by providing hooks to those events via the `gko::log::Logger` abstraction. These hooks include everything from low-level events, such as memory allocations, deallocations, copies and kernel launches, up to high-level events, such as linear operator applications and completions of solver iterations. While the true power of logging is enabled by writing application-specific loggers, Ginkgo does provide several built-in solutions that can be useful for debugging and profiling:

* `gko::log::Convergence` - allows access to the final iteration count and residual of a Krylov solver;
* `gko::log::Stream` - prints events in human-readable format to the given output stream as they are emitted;
* `gko::log::Record` - saves all emitted events in a data structure for subsequent processing;
* `gko::log::Papi` - converts between Ginkgo's logging hooks and the standard PAPI Software Defined Events (SDE) interface (note that some details are lost, as PAPI can represent only a subset of data Ginkgo's logging can provide).

Extensibility
-------------

If you did not find what you need among the built-in components, you can try adding your own implementation of a component. New matrices, solvers and preconditioners can be implemented by inheriting from the `gko::LinOp`
abstract class, while new stopping criteria and loggers by inheriting from the `gko::stop::Criterion` and `gko::log::Logger` abstract classes, respectively. Ginkgo aims at being developer-friendly and provides features that simplify the development of new components. To help handling various memory spaces, there is the `gko::Array` type template that encapsulates memory allocations, deallocations and copies between them. Macros and [mixins](https://en.wikipedia.org/wiki/Mixin) (realized via the [C++ CRTP idiom](https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern)) that implement common utilities on Ginkgo's object are also provided, allowing you to focus on the implementation of your algorithm, instead of implementing various utilities required by the interface.

License
-------

Ginkgo is available under the [BSD 3-clause license](https://github.com/ginkgo-project/ginkgo/blob/v1.0.0/LICENSE). Optional third-party tools and libraries needed to run the unit tests, benchmarks, and developer tools are available under their own open-source licenses, but a fully functional installation of Ginkgo can be obtained without any of them. Check [ABOUT-LICENSING.md](https://github.com/ginkgo-project/ginkgo/blob/v1.0.0/ABOUT-LICENSING.md) for details.

Getting Started
---------------

To learn how to use Ginkgo, and get ideas for your own projects, take a look at the following examples:

* [`minimal-solver-cuda`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/minimal-cuda-solver) is probably one of the smallest complete programs you can write in Ginkgo, and can be used as a quick reference for assembling Ginkgo's components.
* [`simple-solver`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/simple-solver) is a slightly more complex example that reads the matrices from files, computes the final residual, and selects a different executor based on the command-line parameter.
* [`preconditioned-solver`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/preconditioned-solver) is a slightly modified `simple-solver` example that adds that demonstrates how a solver can be enhanced with a preconditioner.
* [`simple-solver-logging`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/simple-solver-logging) is yet another modification of the `simple-solver` example that prints information about the solution process to the screen by using built-in loggers.
* [`poisson-solver`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/poisson-solver) is a more elaborate example that builds a small application for the solution of the 1D Poisson equation using Ginkgo.
* [`three-pt-stencil-solver`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/three-pt-stencil-solver) is a variation of the `poisson_solver` that demonstrates how one could use Ginkgo with software that was not originally designed with Ginkgo support. It encapsulates everything related to Ginkgo in a single function that accepts raw data of the problem and demonstrates how such data can be directly used with Ginkgo's components.
* [`inverse-iteration`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/inverse-iteration) is another full application that uses Ginkgo's solver as a component for implementing the inverse iteration eigensolver.

You can also check out Ginkgo's [core](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/core/test) and [reference](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/reference/test) unit tests and [benchmarks](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/benchmark) for more detailed examples of using each of the components. A complete Doxygen-generated reference is available [online](https://ginkgo-project.github.io/ginkgo-generated-documentation/doc/v1.0.0/), or you can find the same information by directly browsing Ginkgo's [headers](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/include/ginkgo). We are investing significant efforts in maintaining good code quality, so you should not find them difficult to read and understand.

If you want to use your own functionality with Ginkgo, these examples are the best way to start:

* [`custom-logger`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/custom-logger) demonstrates how Ginkgo's logging API can be leveraged to implement application-specific callbacks for Ginkgo's events.
* [`custom-stopping-criterion`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/custom-stopping-criterion) creates a custom stopping criterion that controls when the solver is stopped from another execution thread.
* [`custom-matrix-format`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/custom-matrix-format) demonstrates how new linear operators can be created, by modifying the `poisson-solver` example to use a more efficient matrix format designed specifically for this application.

Ginkgo's [sources](https://github.com/ginkgo-project/ginkgo) can also serve as a good example, since built-in components are mostly implemented using publicly available utilities.

Contributing
------------

Our principal goal for the development of Ginkgo is to provide high quality software to researchers in HPC, and to application scientists that are interested in using this software. We believe that by investing more effort in the initial development of production-ready method, the entire scientific community benefits in the long run. HPC researchers can save time by using Ginkgo's components as a starting point for their algorithms, or to compare Ginkgo's implementations with their own methods. Since Ginkgo is used for bleeding-edge research, application scientists immediately get access to production-ready new methods that help solve their problems more efficiently.

Thus, if you are interested in making this project even better, we would love to hear from you:

* If you have any questions, comments, suggestions, problems, or think you have found a bug, do not hesitate to [post an issue](https://github.com/ginkgo-project/ginkgo/issues/new) (you will have to register on GitHub first to be able to do it). In case you _really_ do not want your comment to be publicly available, you can send us an e-mail to ginkgo.library@gmail.com.
* If you developed, or would like to develop your own component that you think could be useful to others, we would be glad to accept a [pull request](https://github.com/ginkgo-project/ginkgo/pulls) and distribute your component as part of Ginkgo. The community will benefit by having the new method easily available, and you would get the chance to improve your code further as part of the review process with our development team. You may also want to consider creating writing an issue or sending an e-mail about the feature you are trying to implement before you get started for tips on how to best realize it in Ginkgo, and avoid going the wrong way.
* If you just like Ginkgo and want to help, but do not have a specific project in mind, fell free to take on one of the [open issues](https://github.com/ginkgo-project/ginkgo/issues), or send us an issue or an e-mail describing your interests and background and we will find a project you could work on.

Backward Compatibility Guarantee and Future Support
---------------------------------------------------

This is a major **1.0.0** release of Ginkgo. All future patch releases of the form **1.0.x** are guaranteed to keep exactly the same interface as the major release. All minor releases of the form **1.x.y** are guaranteed not to change existing interfaces, but only add new capabilities.

Thus, all code conforming to the **1.0.0** release will continue to compile and run on all future Ginkgo versions up to (but not including) version **2.0.0**.

About
-----

Ginkgo 1.0.0 is brought to you by:

**Karlsruhe Institute of Technology**, Germany  
**Universitat Jaume I**, Spain  
**University of Tennessee, Knoxville**, US  

These universities, along with various project grants, supported the development team and provided resources needed for the development of Ginkgo.

Ginkgo 1.0.0 contains contributions from:

**Hartwig Anzt**, Karlsruhe Institute of Technology  
**Yenchen Chen**, National Taiwan University  
**Terry Cojean**, Karlsruhe Institute of Technology  
**Goran Flegar**, Universitat Jaume I  
**Fritz Göbel**, Karlsruhe Institute of Technology  
**Thomas Grützmacher**, Karlsruhe Institute of Technology  
**Pratik Nayak**, Karlsruhe Institute of Technology  
**Tobias Ribizel**, Karlsruhe Institute of Technology  
**Yuhsiang Tsai**, National Taiwan University  

Supporting materials are provided by the following individuals:

**David Rogers** - the Ginkgo logo  
**Frithjof Fleischhammer** - the Ginkgo website  

The development team is grateful to the following individuals for discussions and comments:

**Erik Boman**  
**Jelena Držaić**  
**Mike Heroux**  
**Mark Hoemmen**  
**Timo Heister**  
**Jens Saak**  

