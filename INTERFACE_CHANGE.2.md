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

# Changed Interface

## Clone related function is not longer belonging to PolymorphicObject [#2005](https://github.com/ginkgo-project/ginkgo/pull/2005)
Previously, Ginkgo provides `clone` to any class inherited from PolymorphicObject (or EnablePolymorphicObject).
We decide to move `clone` related function from default requirement to optional feature via `Cloneable` class.
Users can use `as<ConcreteType>(as<Clonable>(pointer)->clone())` or `as<ConcreteType>(as<gko::Clonable>(pointer)->clone(exec))` to have the clone object when the pointer is base type like `LinOp`.
If the object is the concrete type like Dense and Csr, `pointer->clone()` and `pointer->clone(exec)` works the same as previous version.
Moreover, we still provide `gko::clone([exec,] pointer)` to deal with all cases, which return the same type as input but throw an exception when it is not `Cloneable`.
If Users have their own class inherit from something like `EnableLinOp`, users only need to inherit from `LinOp` for apply function and optionally inherit from `EnableCloneable<ConcreteType>` for clone function.
Matrix format, Vector, Partition, and etc are cloneable. Factory, Solvers, and Preconditioner are not cloneable now.
