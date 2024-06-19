# Distributed Solvers

Many of Ginkgo's solver support distributed systems.
For them, solving a distributed system is exactly the same as for a non-distributed system.
Only the solver has to be generated from a `gko::experimental::distributed::Matrix`, and then applied to `gko::experimental::distributed::Vector`s.
If a preconditioner is used, it also has to support distributed systems.
A brief example is shown here:
```c++
std::shared_ptr<gko::experimental::distributed::Matrix<>> A = ...;
std::unique_ptr<gko::experimental::distributed::Vector<>> b = ...;
std::unique_ptr<gko::experimental::distributed::Vector<>> x = ...;

auto solver = gko::solver::Cg<>::build().on(exec)->generate(A);
solver->apply(b, x);
```

## List of Supported Solvers

These solvers are available for distributed systems.

- BiCGStab
- (F)GMRES
- (F)CG
- CGS
- GCR
- IR
- MultiGrid, depends on the parameters

## List of Unsupported Solvers

These solvers are not available for distributed systems.

- BiCG
- CB-GMRES
- Direct
- IDR
- triangular solver