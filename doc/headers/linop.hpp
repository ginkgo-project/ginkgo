/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

/**
 * @defgroup LinOp Linear Operators
 *
 * @brief A module dedicated to the implementation and usage of the Linear
 * operators in Ginkgo.
 *
 * Below we elaborate on one of the most important concepts of Ginkgo, the
 * linear operator. The linear operator (LinOp) is a base class for all linear
 * algebra objects in Ginkgo. The main benefit of having a single base class for
 * the entire collection of linear algebra objects (as opposed to having
 * separate hierarchies for matrices, solvers and preconditioners) is the
 * generality it provides.
 *
 * @section linop_1 Linear operator as a concept.
 *
 * A key observation for providing a unified interface for matrices, solvers,
 * and preconditioners is that the most common operation performed on all of
 * them can be expressed as an application of a linear operator to a vector:
 *
 * +   the sparse matrix-vector product with a matrix \f$A\f$ is a linear
 *     operator application \f$y = Ax\f$;
 * +   the application of a preconditioner is a linear operator application
 *     \f$y = M^{-1}x\f$, where \f$M\f$ is an approximation of the original
 *     system matrix \f$A\f$ (thus a preconditioner represents an "approximate
 *     inverse" operator \f$M^{-1}\f$).
 * +   the system solve \f$Ax = b\f$ can be viewed as linear operator
 *     application
 *     \f$x = A^{-1}b\f$ (it goes without saying that the implementation of
 *     linear system solves does not follow this conceptual idea), so a linear
 *     system solver can be viewed as a representation of the operator
 *     \f$A^{-1}\f$.
 *
 * Consider the following example of the fixed-point iteration
 * routine \f$x_{k+1} = Lx_k + b\f$ :
 *
 * ```cpp
 * std::unique_ptr<matrix::Dense<>> calculate_fixed_point(
 *         int iters, const LinOp *L, const matrix::Dense<> *x0
 *         const matrix::Dense<> *b)
 * {
 *     auto x = gko::clone(x0);
 *     auto tmp = gko::clone(x0);
 *     auto one = Dense<>::create(L->get_executor(), {1.0,});
 *     for (int i = 0; i < iters; ++i) {
 *         L->apply(gko::lend(tmp), gko::lend(x));
 *         x->add_scaled(gko::lend(one), gko::lend(b));
 *         tmp->copy_from(gko::lend(x));
 *     }
 *     return x;
 * }
 * ```
 *
 * Here, if \f$L\f$ is a matrix, LinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes \f$b = L \cdot a\f$.
 * `x->add_scaled(one.get(), b.get())` is the `axpy` vector update \f$x:=x+b\f$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. This routine is made a part of the LinOp base class and hence
 * the fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 *
 * @section linop_2 How does it work ?.
 *
 * And the answer is runtime polymorphism. Each of the classes that derive from
 * the LinOp class implement an apply method which they override from the base
 * LinOp class. All the various subclasses provide a common interface and hence
 * the library users are exposed to a smaller set of routines. For example, a
 * matrix-vector product, a preconditioner application, or even a system solve
 * are just different terms given to the operation of applying a certain linear
 * operator to a vector. As such, Ginkgo uses the same routine name,
 * LinOp::apply() for each of these operations, where the actual
 * operation performed depends on the type of linear operator involved in
 * the operation.
 *
 * @section linop_3 Advantages of this approach and usage
 *
 * A common interface often allows for writing more generic code. If a
 * user's routine requires only operations provided by the LinOp interface,
 * the same code can be used for any kind of linear operators, independent of
 * whether these are matrices, solvers or preconditioners. This feature is also
 * extensively used in Ginkgo itself. For example, a preconditioner used
 * inside a Krylov solver is a LinOp. This allows the user to supply a wide
 * variety of preconditioners: either the ones which were designed to be used
 * in this scenario (like ILU or block-Jacobi), a user-supplied matrix which is
 * known to be a good preconditioner for the specific problem,
 * or even another solver (e.g., if constructing a flexible GMRES solver).
 *
 * For example, a matrix free implementation would require the user to provide
 * an apply implementation and instead of passing the generated matrix to the
 * solver, they would have to provide their apply implementation for all the
 * executors needed and not other code needs to be changed. See @ref
 * custom_matrix_format example for more details.
 */
