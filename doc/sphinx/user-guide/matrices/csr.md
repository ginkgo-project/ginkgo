# `gko::matrix::Csr`

- storage layout
- with image

## Creating

- reading in matrix data
- from existing arrays
- wrapping user data, similar to vector, refer to full api
- do not create uninitialized mtx and fill manually
    - inconsistent state because of srow
    - could set strategy again to trigger rebuiling of srow

## Supported Operations

The `apply` function is the main entry point for operations with a CSR matrix.
It executes different operations, based on the (runtime) type of the arguments.

To describe the different operations the following objects are used:

```c++
matrix::Csr *A, *B, *C;      // matrices
matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
matrix::Dense *alpha, *beta; // scalars of dimension 1x1
matrix::Identity *I;         // identity matrix
```

Computing product with a (multi-)vector, also called SpMV or SpMM:
```c++
A->apply(b, x);               // x = A * b
A->apply(alpha, b, beta, x);  // x = alpha * A * b + beta * x
```

Computing a product between two sparse matrices, also called SpGEMM:
```c++
A->apply(B, C);               // C = A * B
A->apply(alpha, B, beta, C);  // C = alpha * A * B + beta * C
```

Computing an addition of two sparse matrices, also called SpGEAM:
```c++
A->apply(alpha, I, beta, B);  // B = alpha * A + beta * B
```

:::{note}
Both the SpGEMM and SpGEAM operation require the input matrices to be sorted by column index,
otherwise the algorithms will produce incorrect results.
:::

Additionally, a CSR matrix may be scaled row-wise or column-wise, by applying a `Diagonal` matrix to a CSR matrix.
```c++
matrix::Diagonal *D;

D->apply(A, B);   // row scaling B = D * A 
D->rapply(A, B);  // column scaling B = A * D
```

The matrix may also be scaled by a single scalar through the `scale` and `inv_scale` functions:
```c++
A->scale(alpha);      // A = alpha * A
A->inv_scale(aplha);  // A = 1 / alpha * A
```

## Strategies