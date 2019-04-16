This is the main page for the Ginkgo library user documentation. The repository is hosted on [github](https://github.com/ginkgo-project/ginkgo). Documentation on aspects such as the build system, can be found at the @ref install_ginkgo page. The @ref Examples can help you get started with using Ginkgo. 


### Modules

The Ginkgo library can be grouped into [modules](modules.html) and these modules form the basic building blocks of Ginkgo. The modules can be summarized as follows:

*   @ref Executor : Where do you want your code to be executed ?
*   @ref LinOp : What kind of operation do you want Ginkgo to perform ?
    * @ref solvers : Solve a linear system for a given matrix.
    * @ref precond : Precondition a system for a solve. 
    * @ref mat_formats : Perform a sparse matrix vector multiplication with a particular matrix format.
*   @ref log : Monitor your code execution.
*   @ref stop : Manage your iteration stopping criteria.

