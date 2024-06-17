# Arrays

The basic container for contiguous storage is the `array<T>` class.
It is templated by the storage type.
This type should be a [POD type](https://en.cppreference.com/w/cpp/language/classes#POD_class),
which includes for example `double`, `single`, `int`, `std::complex<T>`.

:::{note}
Some functionality is not available for all storage types.
The restrictions will be noted below.
Wrongful usage will only be caught at link-time, where the error might be similar to: `undefined reference to gko::array<T>::fill(T)`.

## Construction

:::{cpp:function} array() noexcept

The default constructor. A default constructed array contains no elements, and is not associated with a `Executor`.
:::

:::{cpp:function} array(std::shared_ptr<const Executor> exec) noexcept

The default constructor with an `Executor`. The array contains no elements, but it's associated with `exec`.
:::

:::{cpp:function} array(std::shared_ptr<const Executor> exec, size_type size)

Creates an array of 
:::


## Ownership

The class supports two modes of ownership over the stored data.
Ownership of the data is defined here as being responsible for freeing the data when it's not used anymore.
Usually, the array owns it's data, meaning the array will free the data, once the array itself is freed.
But it is also possible to create arrays, which do not own their data.
This allows data that is managed outside to be used within Ginkgo.
Non-owning arrays are called views in Ginkgo.

:::{doxygenfunction} gko::make_array_view
:::



:::{doxygenclass} gko::array
:members: is_owning
:members-only:
:::


## Observation

- size
- executor
- data

## Modification

- data access
- filling
- resizing
- setting executor

## Copy & Move



- executor conservation
- cross executor copy/move
- view behavior
- const views
