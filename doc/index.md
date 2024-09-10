# User Guide

This is the main page for the Ginkgo library user documentation. The repository is hosted on [github](https://github.com/ginkgo-project/ginkgo). 
Documentation on aspects such as the build system, can be found at the [install page](using-ginkgo.md). 
The {gko}`Examples` can help you get started with using Ginkgo.

The Ginkgo library can be grouped into {gko}`modules` and these modules form the basic building blocks of Ginkgo. The modules can be summarized as follows:

*   {gko}`Executor` : Where do you want your code to be executed ?
*   {gko}`LinOp` : What kind of operation do you want Ginkgo to perform ?
    * {gko}`solvers` : Solve a linear system for a given matrix.
    * {gko}`precond` : Precondition a linear system. 
    * {gko}`mat_formats` : Perform a sparse matrix vector multiplication with a particular matrix format.
*   {gko}`log` : Monitor your code execution.
*   {gko}`stop` : Manage your iteration stopping criteria.


:::{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide:

% The Examples and API link have to be done in this hacky way, since sphinx doesn't allow
% their full reference syntax in the toctree

Tutorial <https://github.com/ginkgo-project/ginkgo/wiki/Tutorial:-Building-a-Poisson-Solver>
Examples <../_doxygen/usr/Examples.html#https://>
Publications <publications>
contributing
Using Ginkgo <using-ginkgo>
API <../_doxygen/usr/index.html#https://>
:::