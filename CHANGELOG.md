# Changelog

This file may not always be up to date for the unreleased commits. For a
comprehensive list, use the following commands:
```bash
git log --first-parent
```

## Unreleased
### Added
+ [2d3f0318](https://github.com/ginkgo-project/ginkgo/commit/2d3f0318ed9412a3522d12a85b863efad12fd033), [5e744cad](https://github.com/ginkgo-project/ginkgo/commit/5e744cad1ac0a86b58e3a982dfd7fff4123a7ae3), [22e4b07d](https://github.com/ginkgo-project/ginkgo/commit/22e4b07db7642b54e89c026372c9aae7554ff385), [a5d60de9](https://github.com/ginkgo-project/ginkgo/commit/a5d60de994d0d2073d6dfd4b170fccb557ab6663): Code quality tools in the CI system such as IWYU, clang-tidy and sonarqube.
+ [e1ed14da](https://github.com/ginkgo-project/ginkgo/commit/e1ed14dae236cf4880e1aab418ba8b1784cc8c6e): Fully abide to the xSDK compatibility policies. 
+ [0c60deec](https://github.com/ginkgo-project/ginkgo/commit/0c60deec4ce806394fb3287735fad4fb9e7e5c71), [de51ee9a](https://github.com/ginkgo-project/ginkgo/commit/de51ee9a4fbec45d4af99c877a3a49ab94c8cdb5): Two new examples, a 9pt and 27pt stencil.
+ [5e0ca656](https://github.com/ginkgo-project/ginkgo/commit/5e0ca656865f2fa8c35c3470bc6e531c7cf95b66): Benchmark support for cuSPARSE SpMVs. 
+ [2f2f09eb](https://github.com/ginkgo-project/ginkgo/commit/2f2f09eb8e653b2552fe97c997e6729c6a3dbcdc), [ec7918f0](https://github.com/ginkgo-project/ginkgo/commit/ec7918f0a3ddb8084a7b2854d4f4d88dc86a1c11): Benchmark support for conversion between SpMV formats.
+ [c9be4445](https://github.com/ginkgo-project/ginkgo/commit/c9be444527fb985f9646c4ebb1b8fb7b9ef72615), [82e6da60](https://github.com/ginkgo-project/ginkgo/commit/82e6da6022a4a5405ad2b91f0f48ccc2490114cd): CSR conversions to and from Hybrid.
+ [fce8dad4](https://github.com/ginkgo-project/ginkgo/commit/fce8dad411603fa517e56073c47b0582910a0b1a), [a3307f07](https://github.com/ginkgo-project/ginkgo/commit/a3307f0760174f7f8b9d4edf20688fe5e2ff9d7a): New ParILU preconditioner.
+ [75a398fc](https://github.com/ginkgo-project/ginkgo/commit/75a398fc64aaa17e8ab343a84f4d8d8caa3ca662): Support for sorting CSR matrices. See also the ParILU commits.

### Changed
+ [fe58c940](https://github.com/ginkgo-project/ginkgo/commit/fe58c940aa365d1c7434836150c53fdb4832c3ef): Fix the CUDA conversion from CSR and Dense to Sell-P. 
+ [75806c26](https://github.com/ginkgo-project/ginkgo/commit/75806c26ff6af86d2bb436c9b19a6df3d9be76ce), [c6229b80](https://github.com/ginkgo-project/ginkgo/commit/c6229b804e27c4adb02df17af46f925d48f312ff): General fixes to the CI system scripts.
+ [8bf33e0e](https://github.com/ginkgo-project/ginkgo/commit/8bf33e0e3386d0e6a6c41631444deeea627d1d94), [37dfe3b8](https://github.com/ginkgo-project/ginkgo/commit/37dfe3b865a5902a5e395aa424e13190d1bd2c65): Improve CSR->ELL,Hybrid conversions. 
+ [c4f567eb](https://github.com/ginkgo-project/ginkgo/commit/c4f567ebc80b22252c5c5284a00e4d9f86d22e2c): Fix compilation with GCC 6.4.

### Removed

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

* [`custom_logger`](https://github.com/ginkgo-project/ginkgo/tree/v1.0.0/examples/custom-logger) demonstrates how Ginkgo's logging API can be leveraged to implement application-specific callbacks for Ginkgo's events.
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

