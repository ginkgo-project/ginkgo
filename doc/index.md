# User Guide

This is the main page for the Ginkgo library user documentation. The repository is hosted on [github](https://github.com/ginkgo-project/ginkgo). 
Documentation on aspects such as the build system, can be found at the [install page](user-guide/using-ginkgo.md). 
The {doxy}`Examples` can help you get started with using Ginkgo.

The Ginkgo library can be grouped into modules which form the basic building blocks of Ginkgo. The modules can be summarized as follows:

*   {doxy}`Executor` : Where do you want your code to be executed ?
*   {doxy}`LinOp` : What kind of operation do you want Ginkgo to perform ?
    * {doxy}`solvers` : Solve a linear system for a given matrix.
    * {doxy}`precond` : Precondition a linear system. 
    * {doxy}`mat_formats` : Perform a sparse matrix vector multiplication with a particular matrix format.
*   {doxy}`log` : Monitor your code execution.
*   {doxy}`stop` : Manage your iteration stopping criteria.


% The Examples and API link have to be done in this hacky way, since sphinx doesn't allow
% their full reference syntax in the toctree
:::{toctree}
:maxdepth: 2
:hidden:
:caption: User Guide:

Tutorial <https://github.com/ginkgo-project/ginkgo/wiki/Tutorial:-Building-a-Poisson-Solver>
Examples <_doxygen/html/Examples.html#https://>
Publications <user-guide/publications>
user-guide/contributing
Using Ginkgo <user-guide/using-ginkgo>
API <_doxygen/html/index.html#https://>
:::

:::{toctree}
:hidden:
:maxdepth: 2
:caption: Developer Guide:

developer-guide/documentation
:::