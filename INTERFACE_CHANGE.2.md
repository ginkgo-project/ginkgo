This file collects the interface break introduced by Ginkgo 2.0
If we still support the same function in different way, we will provide how to change the usage.

# Removed Interface

## LinOp chain apply [#2001](https://github.com/ginkgo-project/ginkgo/pull/2001)

`LinOp->apply(b1, x1)->apply(b2, x2);` is not supported. We also drop this chain support for other kinds of apply such as `Coo->apply2(b, x)`, `LinOp->apply(alpha, b, beta, x)`, and `BatchLinOp->apply(b, x)`.  
To be clear, we only drop the chain behavior not LinOp apply itself. i.e. LinOp apply is still supported like the following
```
LinOp->apply(b1, x1);
LinOp->apply(b2, x2);
```

## Template Parameter for Solver Type in `preconditioner::Ic` and `preconditioner::Ilu` [#1998](https://github.com/ginkgo-project/ginkgo/pull/1998) 

Providing any non-ValueType template parameter to `preconditioner::Ic` and `preconditioner::Ilu` is not supported anymore.
The first template parameter is now only used to determine the ValueType of the preconditioner.
If you used something similar to:
```c++
auto ic = preconditioner::Ic<solver::LowerTrs<double>>::build().on(exec);
```
you will now have to use:
```c++
auto ic = preconditioner::Ic<double>::build()
  .with_l_solver(solver::LowerTrs<double>::build())
  .on(exec);
```
Note that the `solver::LowerTrs` is the default solver type for `preconditioner::Ic`, i.e. the above can be shortened to
```c++
auto ic = preconditioner::Ic<double>::build().on(exec);
```
You can provide any `LinOpFactory` to `.with_l_solver()`.

The same applies to `preconditioner::Ilu`, except that the solver for both the lower triangular and upper triangular part can be provided.

## Clone is no longer available by default in Ginkgo Classes
Previously, Ginkgo provides `clone` to any class inherited from `PolymorphicObject` (or `EnablePolymorphicObject`).
We decide to move `clone` related function from default requirement to optional feature via `Cloneable` class. (see changed interface for more details)
That means `obj->clone(...)` might throw compile-time error now when the class does not support `clone`.

Except for the matrix, vector, or utilities like partition, the classes under solver/preconditioner/factorization... do not have the clone now.
If users previously rely on calling `source->clone([exec,] target)`, please use `auto target = gko::clone([exec,] source)`. 
`gko::clone` can adapt whether the class has clone natively or is `Cloneable`. If no clone is supported, it will throw an runtime exception.

Factory does not support clone, either. 
Users, who want to create the same factory for different executors, please create a factory without specifying executor in the components' factory.

## Remove HWLOC related stuff [#2060](https://github.com/ginkgo-project/ginkgo/pull/2060)
It removes the HWLOC usage in Ginkgo, `struct machine_topology`, `get_closest_pus()`,  and `get_closest_numa()`.

# Changed Interface
## Clone related function is not longer belonging to PolymorphicObject [#2005](https://github.com/ginkgo-project/ginkgo/pull/2005)
For those classes supporting clone, the clone is still available under concreate type or via `Cloneable` class.
Users can use `as<ConcreteType>(as<Cloneable>(pointer)->clone())` or `as<ConcreteType>(as<gko::Cloneable>(pointer)->clone(exec))` to have the clone object when the pointer is base type like `LinOp`.
If the object is the concrete type like Dense and Csr, `pointer->clone()` and `pointer->clone(exec)` works the same as previous version.

If Users have their own class inherit from something like `EnableLinOp`, users only need to inherit from `LinOp` for `apply` function and optionally inherit from `EnableCloneable<ConcreteType>` for `clone` function.

## Csr create function with the strategy
No csr strategy with the class inheritence.
`std::make_shared<Csr::<strategy_type>>(...)` -> `gko::matrix::csr::spmv_strategy::<strategy_name>`
(currently still allow basic usage on the strategy shared pointer with deprecation warning).  
`csr->get_strategy()` return enum class not shared_ptr.  
No manual setup for load_balance strategy. (If you need it, please let us know)
