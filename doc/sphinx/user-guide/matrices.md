# Matrices


- matrices created with `gko::matrix::<format>::create(...)`
- sets executor
- may provide dimensions
- additional settings, e.g. Csr strategy, or max number of elements per row
- fill matrix entries with `read`
  - dimensions set by create will be overriden 
  - ref [](matrices/matrix_data.md)
- alternative provide already filled arrays to `create`
  - arrays depend on format
- matrices can be applied to vectors by `apply`

```c++
std::unique_ptr<gko::matrix::Dense<>> b = ...;
std::unique_ptr<gko::matrix::Dense<>> x = ...;
std::unique_ptr<gko::matrix::Dense<>> alpha = ...; // 1x1 vector
std::unique_ptr<gko::matrix::Dense<>> beta = ...; // 1x1 vector
gko::matrix_data<> md(...);

auto A = gko::matrix::Csr<>::create(exec);
A->read(md);

A->apply(b, x);               // x = A * b
A->apply(alpha, b, beta, x);  // x = alpha * A * b + beta * x
```

- dimension checking in apply

```c++
auto b = gko::matrix::Dense<>::create(exec, gko::dim<2>{3, 2});
auto x = gko::matrix::Dense<>::create(exec, gko::dim<2>{5, 2});
auto A = gko::matrix::Csr<>::create(exec, gko::dim<2>{5, 3});
A->apply(b, x);   // works

auto b2 = gko::matrix::Dense<>::create(exec, gko::dim<2>{2, 3});
A->apply(b2, x);  // throws exception
```

## Matrix Formats

The format with the most features is [](matrices/csr.md).
Almost all other formats can be converted to and from Csr.
Many algorithms also require their input as Csr matrices, so if a matrix in a different format is provided, 
it might get converted to Csr beforehand.

:::{toctree}

matrices/coo
matrices/csr
matrices/diagonal
matrices/ell
matrices/fbcsr
matrices/hybrid
matrices/identity
matrices/permutation
matrices/row_gatherer
matrices/scaled_permutation
matrices/sellp
matrices/sparsity

:::