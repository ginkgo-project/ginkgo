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
# Changed Interface
