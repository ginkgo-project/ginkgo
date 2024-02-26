// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
 * executors needed and no other code needs to be changed. See @ref
 * custom_matrix_format example for more details.
 */
