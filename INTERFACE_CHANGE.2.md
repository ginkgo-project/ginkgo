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
