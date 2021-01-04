# Changelog

This file may not always be up to date in particular for the unreleased
commits. For a comprehensive list, use the following command:
```bash
git log --first-parent
```

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
+ Support for sorting rows in the CSR format by column idices ([#322](https://github.com/ginkgo-project/ginkgo/issues/322))
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

You can also check out Ginkgo's [core](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/core/test) and [reference](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/reference/test) unit tests and [benchmarks](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/benchmark) for more detailed examples of using each of the components. A complete Doxygen-generated reference is available [online](https://ginkgo-project.github.io/ginkgo/doc/v1.0.0/), or you can find the same information by directly browsing Ginkgo's [headers](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/include/ginkgo). We are investing significant efforts in maintaining good code quality, so you should not find them difficult to read and understand.

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
**Pratik Nayak**, Karlsruhe Institue of Technologgy  
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

