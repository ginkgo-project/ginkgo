### Modules

The Ginkgo library can be grouped into [modules](modules.html) and these modules form the basic building blocks of Ginkgo. The modules can be summarized as follows:

\dotfile modules.dot

*   @ref Executor : Where do you want your code to be executed ?
*   @ref LinOp : What kind of operation do you want Ginkgo to perform ?
    * @ref solvers : Solve a linear system for a given matrix.
    * @ref precond : Precondition a system for a solve. 
    * @ref mat_formats : Perform a sparse matrix vector multiplication with a particular matrix format.
*   @ref log : Monitor your code execution.
*   @ref stop : Manage your iteration stopping criteria.

