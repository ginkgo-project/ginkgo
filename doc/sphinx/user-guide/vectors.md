# Vectors

Sequential vectors of $\mathbb{R}^n$ or $\mathbb{C}^n$ are represented by `gko::matrix::Dense<T>`.


Objects of type `gko::matrix::Dense<T>` are managed through smart pointers.
These behave like 'normal' pointers, except that they automatically manage the lifetime of the object.
For more details on how Ginkgo uses smart pointers please refer to [](lifetime).

- row-major storage format


## Creating

- create functions
- with stride
- ref [](vectors/dim.md)
- ref [](vectors/user-data.md)

## Accessing Data

## BLAS Operations

- operations typical of elements of vector spaces
- norms
  - multiple columns 
- transpose

## Conversions

- precision conversions
- to matrix conversions
- to/from complex 

## IO

- read/write from/to matrix data [](matrices/matrix_data.md)
- mtx format for multiple columns is a single list, which has all entries in column major order

## Permutation

## Gathering Rows

## Submatrices

:::{toctree}
:hidden:

vectors/dim
vectors/user-data
:::