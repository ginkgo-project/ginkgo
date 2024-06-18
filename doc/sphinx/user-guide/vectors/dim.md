# `gko::dim<N>`

The extent of an `N`-dimensional tensor is represented by a `gko::dim<N>` object.
In nearly all instances `N` is exclusively set to `N=2`, which corresponds to elements from $\mathbb{R}^{n\times m}$.

A `gko::dim<N>` object is created by passing in the extents for each dimensions.
For `N=2` this corresponds to passing in the number of rows and number of columns of a matrix.

```c++
gko::size_type n1, n2, n3, n4, num_rows, num_cols;
gko::dim<4> dim4(n1, n2, n3, n4);
gko::dim<2> size(num_rows, num_cols);
```

The `gko::dim<N>` provides access to the extents through the `[]` operator.
The extent of the i-th dimension (starting with zero) is given by `size[i]`.

```c++
gko::dim<2> size(num_rows, num_cols);
auto extent = size[1];
assert(extent == num_cols);
```

A `gko::dim<N>` object can be converted to bool.
It evaluates to `true` if and only if *all* extents are larger than zero.
As soon as any extent is zero, it will evaluate to `false`.
This allows using `gko::dim<N>` in `if` expression, perhaps to catch edge cases.

```c++
gko::dim<2> size(0, 0);
if(!size){
  // handle empty size case
}
```

The `gko::dim<N>` class also supports equality checking of two objects.
They are the same if and only if all their extents are the same.