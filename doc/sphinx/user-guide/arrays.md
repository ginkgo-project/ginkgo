# `gko::array`

The basic container for contiguous storage is the `array<T>` class.
It is templated by the storage type.
This type should be built-in type, or a struct of built-in types,
which includes for example `double`, `single`, `int`, `std::complex<T>`.
Such a type is often referred to as  a [POD type](https://en.cppreference.com/w/cpp/language/classes#POD_class).
The type **may not** be const-qualified.
Handling constant data is done through const-qualifying the whole array.

:::{attention}
Some functionality is not available for all storage types.
The restrictions will be noted below.
Wrongful usage will only be caught at link-time, where the error might be similar to: `undefined reference to gko::array<T>::fill(T)`.
:::

## Creating

An array is constructed from an `Executor` and the number of stored objects:

```cpp
std::shared_ptr<const gko::Executor> exec = ...;
gko::size_type size = ...;
gko::array<double> a(exec, size);
```

:::{attention}
After construction, the stored memory of the array is *not* initialized.
:::


To create an array that is already filled with data, an initializer list can be passed instead of the size parameter.
An array with the elements `[1.0, 2.0, 3.0]` can be created as shown in the following.
Note that although the initializer list lives in CPU memory, the created array will put the elements into the memory
space defined by the executor.

```c++
gko::array<double> b(exec, {1.0, 2.0, 3.0});
```

An alternative is to use begin and end iterators from a container with CPU data.
The data from the container is then copied into the array.

```c++
std::vector std_c{1.0, 2.0, 3.0};
gko::array<double> c(exec, std_c.begin(), std_c.end());
```

It is also possible to default create an array:

```c++
gko::array<double> d;
gko::array<double> e(exec);
```

The first array `d` stores zero elements, and it is not associated with any executor.
The second array `e` also stores zero elements, but it is associated with the executor `exec`.
The difference between both is mostly relevant for [copy and move operations](#copy-move).

It is also possible to create an array that doesn't allocate data, and instead takes existing data as input.
More details are in [](arrays/ownership.md).

## Using

The main access to the stored data of an array is through the functions `get_data`, and `get_const_data`:

```c++
double* a_ptr = a.get_data();
const double* a_const_ptr = a.get_const_data();
```

The `get_data` function is only available on non-const arrays.
If the array has been default allocated, or the size is zero, then the returned pointers will be `nullptr`.

:::{note}
The pattern of having two accessors `get_<something>` and `get_const_<something>`,
one for mutable and one for const access respectively, is very common for Ginkgo types.
For `const` objects, this means that only the `get_const_<something>` accessor will be available,
making mutable access (nearly) impossible.
:::

:::{warning}
The memory addresses returned by both functions will be on the backend defined by the executor.
For example, if the array was created with an `CudaExecutor`, then the memory will be on the GPU device.
Thus the memory can only be accessed on the device.
:::

The number of stored elements can be queried with `get_size`:

```c++
gko::array<double> s(exec, 10);
gko::size_type size = s.get_size();
assert(size == 10);
```

It is also possible to get back the `Executor` used to create an array with `get_executor`.
In case of a default constructed array without executor, this will return a `nullptr`.

```c++
std::shared_ptr<const gko::Executor> exec = a.get_executor();
```

The whole array might be set to a single value with `fill(T value)`.

```c++
gko::array<double> a(exec, 10);
a.fill(1.1);
```

Since this operation might require kernel calls depending on the backend, it is not available for any storage type.
The supported types are:

- real and complex floating point types, i.e. `float`, `double`, `std::complex<float>`, and `std::complex<double>`
- index and size types, i.e. `int32`, `int64`, and `size_type`

So the following would result in an error at link-time:

```c++
class POD{
  int i;
};

gko::array<POD> pod_arr(exec, 10);
pod_arr.fill(POD{4});  //< this will result in a link-time error
```

### Resetting and Resizing

The array may be resized with the function `resize_and_reset(gko::size_type size)`.
As the name suggest, the stored content will be reset.
If the `size` is the same as the current number of stored, the array will not be changed.

```c++
a.resize_and_reset(20);
assert(a.get_size() == 20);
```

:::{admonition} Exceptions
:class: warning

This throws an exception if the array is not associated with an executor, e.g. if it was default constructed.
It will also throw if the array is non-owning, see [](#ownership). 
:::

To set the number of elements to zero, the function `clear` can be used.
Afterward, the accessors `get_data` and `get_const_data` will return `nullptr`.

```c++
a.clear();
assert(a.get_size() == 0);
```

It is also possible to change the associated executor with `set_executor(std::shared_ptr<const Executor> exec)`.
If the new executor is different from the current executor, the data will be copied over to the memory space defined by the new executor.

```c++
gko::array<double> a(omp_exec, {1.0, 2.0, 3.0});
a.set_executor(hip_exec);  //< the values [1.0, 2.0, 3.0] are now in GPU memory
```

(copy-move)=
## Copy & Move


It is important to note that a copy or move assignment will **never** change the executor of either array.
This allows an convenient approach to cross-device memory movement.

- executor conservation
- cross executor copy/move
- view behavior
- const views


:::{toctree}
:hidden:

arrays/ownership
:::