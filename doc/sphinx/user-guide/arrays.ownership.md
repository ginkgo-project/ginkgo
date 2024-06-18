# Ownership

The array class supports two modes of ownership over the stored data.
Ownership of the data is defined here as being responsible for freeing the data when it's not used anymore.
Usually, the array owns it's data, meaning the array will free the data, once the array itself is freed.
But it is also possible to create arrays, which do not own their data.
This allows data that is managed outside to be used within Ginkgo.
Non-owning arrays are called views in Ginkgo.

A non-owning array can be created with the helper function `make_array_view<T>(std::shared_ptr<const Executor> exec, size_type size, T* data)`.
The memory `data` is *required* to be accessible by the executor `exec`.
Also, the size must be less or equal to the number of elements allocated at `data`.

```c++
std::shared_ptr<const Executor> exec = ...;
gko::size_type size = ...;
double* data = ...;  //< has to be accessible by exec
gko::array<double> view = gko::make_array_view(exec, size, data);
```

The above function only works with pointers to mutable data.
If a view on const data needs to be created, then the function `make_const_array_view` has to be used.
This will create an 'array-like' view object, which can then either copied into a mutable array, or passed into Ginkgo functions that can handle this type.
Typically this will be `create_const` functions which the Ginkgo matrix types provide.

```c++
std::shared_ptr<const Executor> exec = ...;
gko::size_type size = ...;
const double* data = ...;
auto view = gko::make_const_array_view(exec, size, data);  //< the returned type is an implementation detail 
                                                           //< and should not be relied upon
```

A view can also be constructed from an existing array with `as_view()` and `as_const_view()`.
It doesn't matter in that case if the original array is owning or not.

```c++
std::array<double> a(exec, 10);
auto view = a.as_view();
```

The owning status of an array can be queried with `is_owning()`.
For owning arrays it returns `true` and for non-owning `false`.

```c++
auto owning = gko::array<double>(exec, 5);
auto non_owning = gko::make_array_view(exec, 5, data);
assert(owning.is_owning() == true);
assert(non_owning.is_owning() == false);
```

:::{warning}
A non-owning array shouldn't use the `reset_and_resize` function.
It will throw, if the new size is different from the current size.
:::
