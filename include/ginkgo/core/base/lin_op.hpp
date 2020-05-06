/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_LIN_OP_HPP_
#define GKO_CORE_BASE_LIN_OP_HPP_


#include <memory>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {


/**
 * @addtogroup LinOp
 *
 * @section linop_concept Linear operator as a concept
 *
 * The linear operator (LinOp) is a base class for all linear algebra objects
 * in Ginkgo. The main benefit of having a single base class for the
 * entire collection of linear algebra objects (as opposed to having separate
 * hierarchies for matrices, solvers and preconditioners) is the generality
 * it provides.
 *
 * First, since all subclasses provide a common interface, the library users are
 * exposed to a smaller set of routines. For example, a
 * matrix-vector product, a preconditioner application, or even a system solve
 * are just different terms given to the operation of applying a certain linear
 * operator to a vector. As such, Ginkgo uses the same routine name,
 * LinOp::apply() for each of these operations, where the actual
 * operation performed depends on the type of linear operator involved in
 * the operation.
 *
 * Second, a common interface often allows for writing more generic code. If a
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
 * A key observation for providing a unified interface for matrices, solvers,
 * and preconditioners is that the most common operation performed on all of
 * them can be expressed as an application of a linear operator to a vector:
 *
 * +   the sparse matrix-vector product with a matrix $A$ is a linear
 *     operator application $y = Ax$;
 * +   the application of a preconditioner is a linear operator application
 *     $y = M^{-1}x$, where $M$ is an approximation of the original
 *     system matrix $A$ (thus a preconditioner represents an "approximate
 *     inverse" operator $M^{-1}$).
 * +   the system solve $Ax = b$ can be viewed as linear operator
 *     application
 *     $x = A^{-1}b$ (it goes without saying that the implementation of
 *     linear system solves does not follow this conceptual idea), so a linear
 *     system solver can be viewed as a representation of the operator
 *     $A^{-1}$.
 *
 * Finally, direct manipulation of LinOp objects is rarely required in
 * simple scenarios. As an illustrative example, one could construct a
 * fixed-point iteration routine $x_{k+1} = Lx_k + b$ as follows:
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
 * Here, if $L$ is a matrix, LinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes $b = L \cdot a$.
 * `x->add_scaled(one.get(), b.get())` is the `axpy` vector update $x:=x+b$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. Since this routine is part of the LinOp base class, the
 * fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 *
 * @ref LinOp
 */
class LinOp : public EnableAbstractPolymorphicObject<LinOp> {
public:
    /**
     * Applies a linear operator to a vector (or a sequence of vectors).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     *
     * @return this
     */
    LinOp *apply(const LinOp *b, LinOp *x)
    {
        this->template log<log::Logger::linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b, x);
        return this;
    }

    /**
     * @copydoc apply(const LinOp *, LinOp *)
     */
    const LinOp *apply(const LinOp *b, LinOp *x) const
    {
        this->template log<log::Logger::linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b, x);
        return this;
    }

    /**
     * Performs the operation x = alpha * op(b) + beta * x.
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     *
     * @return this
     */
    LinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                 LinOp *x)
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return this;
    }

    /**
     * @copydoc apply(const LinOp *, const LinOp *, const LinOp *, LinOp *)
     */
    const LinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                       LinOp *x) const
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return this;
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    const dim<2> &get_size() const noexcept { return size_; }

    /**
     * Returns true if the linear operator uses the data given in x as
     * an initial guess. Returns false otherwise.
     *
     * @return true if the linear operator uses the data given in x as
     *         an initial guess. Returns false otherwise.
     */
    virtual bool apply_uses_initial_guess() const { return false; }

protected:
    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit LinOp(std::shared_ptr<const Executor> exec,
                   const dim<2> &size = dim<2>{})
        : EnableAbstractPolymorphicObject<LinOp>(exec), size_{size}
    {}

    /**
     * Sets the size of the operator.
     *
     * @param value  the new size of the operator
     */
    void set_size(const dim<2> &value) noexcept { size_ = value; }

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, LinOp *).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void apply_impl(const LinOp *b, LinOp *x) const = 0;

    /**
     * Implementers of LinOp should override this function instead
     * of apply(const LinOp *, const LinOp *, const LinOp *, LinOp *).
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    virtual void apply_impl(const LinOp *alpha, const LinOp *b,
                            const LinOp *beta, LinOp *x) const = 0;

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    void validate_application_parameters(const LinOp *b, const LinOp *x) const
    {
        GKO_ASSERT_CONFORMANT(this, b);
        GKO_ASSERT_EQUAL_ROWS(this, x);
        GKO_ASSERT_EQUAL_COLS(b, x);
    }

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    void validate_application_parameters(const LinOp *alpha, const LinOp *b,
                                         const LinOp *beta,
                                         const LinOp *x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_EQUAL_DIMENSIONS(alpha, dim<2>(1, 1));
        GKO_ASSERT_EQUAL_DIMENSIONS(beta, dim<2>(1, 1));
    }

private:
    dim<2> size_{};
};


/**
 * A LinOpFactory represents a higher order mapping which transforms one
 * linear operator into another.
 *
 * In Ginkgo, every linear solver is viewed as a mapping. For example,
 * given an s.p.d linear system $Ax = b$, the solution $x = A^{-1}b$
 * can be computed using the CG method. This algorithm can be represented in
 * terms of linear operators and mappings between them as follows:
 *
 * -   A Cg::Factory is a higher order mapping which, given an input operator
 *     $A$, returns a new linear operator $A^{-1}$ stored in "CG
 *     format"
 * -   Storing the operator $A^{-1}$ in "CG format" means that the data
 *     structure used to store the operator is just a simple pointer to the
 *     original matrix $A$. The application $x = A^{-1}b$ of such an
 *     operator can then be implemented by solving the linear system
 *     $Ax = b$ using the CG method. This is achieved in code by having a
 *     special class for each of those "formats" (e.g. the "Cg" class defines
 *     such a format for the CG solver).
 *
 * Another example of a LinOpFactory is a preconditioner. A preconditioner for
 * a linear operator $A$ is a linear operator $M^{-1}$, which
 * approximates $A^{-1}$. In addition, it is stored in a way such that
 * both the data of $M^{-1}$ is cheap to compute from $A$, and the
 * operation $x = M^{-1}b$ can be computed quickly. These operators are
 * useful to accelerate the convergence of  Krylov solvers.
 * Thus, a preconditioner also fits into the LinOpFactory framework:
 *
 * -   The factory maps a linear operator $A$ into a preconditioner
 *     $M^{-1}$ which is stored in suitable format (e.g. as a product of
 *     two factors in case of ILU preconditioners).
 * -   The resulting linear operator implements the application operation
 *     $x = M^{-1}b$ depending on the format the preconditioner is stored
 *     in (e.g. as two triangular solves in case of ILU)
 *
 * Example: using CG in Ginkgo
 * ---------------------------
 *
 * ```c++
 * // Suppose A is a matrix, b a rhs vector, and x an initial guess
 * // Create a CG which runs for at most 1000 iterations, and stops after
 * // reducing the residual norm by 6 orders of magnitude
 * auto cg_factory = solver::Cg<>::build()
 *     .with_max_iters(1000)
 *     .with_rel_residual_goal(1e-6)
 *     .on(cuda);
 * // create a linear operator which represents the solver
 * auto cg = cg_factory->generate(A);
 * // solve the system
 * cg->apply(gko::lend(b), gko::lend(x));
 * ```
 *
 * @ingroup LinOp
 */
class LinOpFactory
    : public AbstractFactory<LinOp, std::shared_ptr<const LinOp>> {
public:
    using AbstractFactory<LinOp, std::shared_ptr<const LinOp>>::AbstractFactory;

    std::unique_ptr<LinOp> generate(std::shared_ptr<const LinOp> input) const
    {
        this->template log<log::Logger::linop_factory_generate_started>(
            this, input.get());
        auto generated = AbstractFactory::generate(input);
        this->template log<log::Logger::linop_factory_generate_completed>(
            this, input.get(), generated.get());
        return generated;
    }
};


/**
 * Linear operators which support transposition should implement the
 * Transposable interface.
 *
 * It provides two functionalities, the normal transpose and the
 * conjugate transpose.
 *
 * The normal transpose returns the transpose of the linear operator without
 * changing any of its elements representing the operation, $B = A^{T}$.
 *
 * The conjugate transpose returns the conjugate of each of the elements and
 * additionally transposes the linear operator representing the operation, $B
 * = A^{H}$.
 *
 * Example: Transposing a Csr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Transposing an object of LinOp type.
 * //The object you want to transpose.
 * auto op = matrix::Csr::create(exec);
 * //Transpose the object by first converting it to a transposable type.
 * auto trans = op->transpose();
 * ```
 */
class Transposable {
public:
    virtual ~Transposable() = default;

    /**
     * Returns a LinOp representing the transpose of the Transposable object.
     *
     * @return a pointer to the new transposed object
     */
    virtual std::unique_ptr<LinOp> transpose() const = 0;

    /**
     * Returns a LinOp representing the conjugate transpose of the Transposable
     * object.
     *
     * @return a pointer to the new conjugate transposed object
     */
    virtual std::unique_ptr<LinOp> conj_transpose() const = 0;
};


/**
 * Linear operators which support permutation should implement the
 * Permutable interface.
 *
 * It provides four functionalities, the row permute, the
 * column permute, the inverse row permute and the inverse column permute.
 *
 * The row permute returns the permutation of the linear operator after
 * permuting the rows of the linear operator. For example, if for a matrix A,
 * the permuted matrix A' and the permutation array perm, the row i of the
 * matrix A is the row perm[i] in the matrix A'. And similarly, for the inverse
 * permutation, the row i in the matrix A' is the row perm[i] in the matrix A.
 *
 * The column permute returns the permutation of the linear operator after
 * permuting the columns of the linear operator. The definitions of permute and
 * inverse permute for the row_permute hold here as well.
 *
 * Example: Permuting a Csr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Permuting an object of LinOp type.
 * //The object you want to permute.
 * auto op = matrix::Csr::create(exec);
 * //Permute the object by first converting it to a Permutable type.
 * auto perm = op->row_permute(permutation_indices);
 * ```
 */
template <typename IndexType>
class Permutable {
public:
    virtual ~Permutable() = default;

    /**
     * Returns a LinOp representing the row permutation of the Permutable
     * object.
     *
     * @param permutation_indices  the array of indices contaning the
     * permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<LinOp> row_permute(
        const Array<IndexType> *permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the column permutation of the Permutable
     * object.
     *
     * @param permutation_indices  the array of indices contaning the
     * permutation order.
     *
     * @return a pointer to the new column permuted object
     */
    virtual std::unique_ptr<LinOp> column_permute(
        const Array<IndexType> *permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the row permutation of the inverse permuted
     * object.
     *
     * @param inverse_permutation_indices  the array of indices contaning the
     * inverse permutation order.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<LinOp> inverse_row_permute(
        const Array<IndexType> *inverse_permutation_indices) const = 0;

    /**
     * Returns a LinOp representing the row permutation of the inverse permuted
     * object.
     *
     * @param inverse_permutation_indices  the array of indices contaning the
     * inverse permutation order.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<LinOp> inverse_column_permute(
        const Array<IndexType> *inverse_permutation_indices) const = 0;
};


/**
 * A LinOp implementing this interface can read its data from a matrix_data
 * structure.
 *
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class ReadableFromMatrixData {
public:
    virtual ~ReadableFromMatrixData() = default;

    /**
     * Reads a matrix from a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void read(const matrix_data<ValueType, IndexType> &data) = 0;
};


/**
 * A LinOp implementing this interface can write its data to a matrix_data
 * structure.
 *
 * @ingroup LinOp
 */
template <typename ValueType, typename IndexType>
class WritableToMatrixData {
public:
    virtual ~WritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(matrix_data<ValueType, IndexType> &data) const = 0;
};


/**
 * A LinOp implementing this interface can be preconditioned.
 *
 * @ingroup precond
 * @ingroup LinOp
 */
class Preconditionable {
public:
    virtual ~Preconditionable() = default;

    /**
     * Returns the preconditioner operator used by the Preconditionable.
     *
     * @return the preconditioner operator used by the Preconditionable
     */
    virtual std::shared_ptr<const LinOp> get_preconditioner() const
    {
        return preconditioner_;
    }

    /**
     * Sets the preconditioner operator used by the Preconditionable.
     *
     * @param new_precond  the new preconditioner operator used by the
     *                     Preconditionable
     */
    virtual void set_preconditioner(std::shared_ptr<const LinOp> new_precond)
    {
        preconditioner_ = new_precond;
    }

private:
    std::shared_ptr<const LinOp> preconditioner_{};
};


/**
 * The EnableLinOp mixin can be used to provide sensible default implementations
 * of the majority of the LinOp and PolymorphicObject interface.
 *
 * The goal of the mixin is to facilitate the development of new LinOp, by
 * enabling the implementers to focus on the important parts of their operator,
 * while the library takes care of generating the trivial utility functions.
 * The mixin will provide default implementations for the entire
 * PolymorphicObject interface, including a default implementation of
 * `copy_from` between objects of the new LinOp type. It will also hide the
 * default LinOp::apply() methods with versions that preserve the static type of
 * the object.
 *
 * Implementers of new LinOps are required to specify only the following
 * aspects:
 *
 * 1.  Creation of the LinOp: This can be facilitated via either
 *     EnableCreateMethod mixin (used mostly for matrix formats),
 *     or GKO_ENABLE_LIN_OP_FACTORY macro (used for operators created from other
 *     operators, like preconditioners and solvers).
 * 2.  Application of the LinOp: Implementers have to override the two
 *     overloads of the LinOp::apply_impl() virtual methods.
 *
 * @tparam ConcreteLinOp  the concrete LinOp which is being implemented
 *                        [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of LinOp
 *
 * @ingroup LinOp
 */
template <typename ConcreteLinOp, typename PolymorphicBase = LinOp>
class EnableLinOp
    : public EnablePolymorphicObject<ConcreteLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteLinOp> {
public:
    using EnablePolymorphicObject<ConcreteLinOp,
                                  PolymorphicBase>::EnablePolymorphicObject;

    const ConcreteLinOp *apply(const LinOp *b, LinOp *x) const
    {
        this->template log<log::Logger::linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b, x);
        return self();
    }

    ConcreteLinOp *apply(const LinOp *b, LinOp *x)
    {
        this->template log<log::Logger::linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_apply_completed>(this, b, x);
        return self();
    }

    const ConcreteLinOp *apply(const LinOp *alpha, const LinOp *b,
                               const LinOp *beta, LinOp *x) const
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return self();
    }

    ConcreteLinOp *apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                         LinOp *x)
    {
        this->template log<log::Logger::linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteLinOp);
};


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of LinOpFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteLinOp  the concrete LinOp type which this factory produces,
 *                        needs to have a constructor which takes a
 *                        const ConcreteFactory *, and an
 *                        std::shared_ptr<const LinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of LinOpFactory
 *
 * @ingroup LinOp
 */
template <typename ConcreteFactory, typename ConcreteLinOp,
          typename ParametersType, typename PolymorphicBase = LinOpFactory>
using EnableDefaultLinOpFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteLinOp, ParametersType,
                         PolymorphicBase>;

/**
 * This Macro will generate a new type containing the parameters for the factory
 * `_factory_name`. For more details, see #GKO_ENABLE_LIN_OP_FACTORY().
 * It is required to use this macro **before** calling the
 * macro #GKO_ENABLE_LIN_OP_FACTORY().
 * It is also required to use the same names for all parameters between both
 * macros.
 *
 * @param _parameters_name  name of the parameters member in the class
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup LinOp
 */
#define GKO_CREATE_FACTORY_PARAMETERS(_parameters_name, _factory_name) \
public:                                                                \
    class _factory_name;                                               \
    struct _parameters_name##_type                                     \
        : ::gko::enable_parameters_type<_parameters_name##_type,       \
                                        _factory_name>


/**
 * This macro will generate a default implementation of a LinOpFactory for the
 * LinOp subclass it is defined in.
 *
 * It is required to first call the macro #GKO_CREATE_FACTORY_PARAMETERS()
 * before this one in order to instantiate the parameters type first.
 *
 * The list of parameters for the factory should be defined in a code block
 * after the macro definition, and should contain a list of
 * GKO_FACTORY_PARAMETER declarations. The class should provide a constructor
 * with signature
 * _lin_op(const _factory_name *, std::shared_ptr<const LinOp>)
 * which the factory will use a callback to construct the object.
 *
 * A minimal example of a linear operator is the following:
 *
 * ```c++
 * struct MyLinOp : public EnableLinOp<MyLinOp> {
 *     GKO_ENABLE_LIN_OP_FACTORY(MyLinOp, my_parameters, Factory) {
 *         // a factory parameter named "my_value", of type int and default
 *         // value of 5
 *         int GKO_FACTORY_PARAMETER(my_value, 5);
 *     };
 *     // constructor needed by EnableLinOp
 *     explicit MyLinOp(std::shared_ptr<const Executor> exec) {
 *         : EnableLinOp<MyLinOp>(exec) {}
 *     // constructor needed by the factory
 *     explicit MyLinOp(const Factory *factory,
 *                      std::shared_ptr<const LinOp> matrix)
 *         : EnableLinOp<MyLinOp>(factory->get_executor()), matrix->get_size()),
 *           // store factory's parameters locally
 *           my_parameters_{factory->get_parameters()},
 *     {
 *          int value = my_parameters_.my_value;
 *          // do something with value
 *     }
 * ```
 *
 * MyLinOp can then be created as follows:
 *
 * ```c++
 * auto exec = gko::ReferenceExecutor::create();
 * // create a factory with default `my_value` parameter
 * auto fact = MyLinOp::build().on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 5
 *
 * // create a factory with custom `my_value` parameter
 * auto fact = MyLinOp::build().with_my_value(0).on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 0
 * ```
 *
 * @note It is possible to combine both the #GKO_CREATE_FACTORY_PARAMETER()
 * macro with this one in a unique macro for class __templates__ (not with
 * regular classes). Splitting this into two distinct macros allows to use them
 * in all contexts. See <https://stackoverflow.com/q/50202718/9385966> for more
 * details.
 *
 * @param _lin_op  concrete operator for which the factory is to be created
 *                 [CRTP parameter]
 * @param _parameters_name  name of the parameters member in the class
 *                          (its type is `<_parameters_name>_type`, the
 *                          protected member's name is `<_parameters_name>_`,
 *                          and the public getter's name is
 *                          `get_<_parameters_name>()`)
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup LinOp
 */
#define GKO_ENABLE_LIN_OP_FACTORY(_lin_op, _parameters_name, _factory_name)  \
public:                                                                      \
    const _parameters_name##_type &get_##_parameters_name() const            \
    {                                                                        \
        return _parameters_name##_;                                          \
    }                                                                        \
                                                                             \
    class _factory_name                                                      \
        : public ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,    \
                                                  _parameters_name##_type> { \
        friend class ::gko::EnablePolymorphicObject<_factory_name,           \
                                                    ::gko::LinOpFactory>;    \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,  \
                                                   _factory_name>;           \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)  \
            : ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,       \
                                               _parameters_name##_type>(     \
                  std::move(exec))                                           \
        {}                                                                   \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,  \
                               const _parameters_name##_type &parameters)    \
            : ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,       \
                                               _parameters_name##_type>(     \
                  std::move(exec), parameters)                               \
        {}                                                                   \
    };                                                                       \
    friend ::gko::EnableDefaultLinOpFactory<_factory_name, _lin_op,          \
                                            _parameters_name##_type>;        \
                                                                             \
                                                                             \
private:                                                                     \
    _parameters_name##_type _parameters_name##_;                             \
                                                                             \
public:                                                                      \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Defines setters and getters for the parameters of a class, simplifying its
 * construction by removing the repetitive typing of boiler plate code.
 *
 * @param _concrete_parameter_name  the parameter whose setter and getter is to
 * be defined
 *
 * @ingroup LinOp
 */
#define GKO_ENABLE_SET_GET_PARAMETERS(_concrete_parameter_name, _type)       \
public:                                                                      \
    void set_##_concrete_parameter_name(_type other)                         \
    {                                                                        \
        this->_concrete_parameter_name##_ = other;                           \
    }                                                                        \
                                                                             \
    _type get_##_concrete_parameter_name() const                             \
    {                                                                        \
        return this->_concrete_parameter_name##_;                            \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


/**
 * Defines a build method for the factory, simplifying its construction by
 * removing the repetitive typing of factory's name.
 *
 * @param _factory_name  the factory for which to define the method
 *
 * @ingroup LinOp
 */
#define GKO_ENABLE_BUILD_METHOD(_factory_name)                               \
    static auto build()->decltype(_factory_name::create())                   \
    {                                                                        \
        return _factory_name::create();                                      \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#if !(defined(__CUDACC__) || defined(__HIPCC__))
/**
 * Creates a factory parameter in the factory parameters structure.
 *
 * @param _name  name of the parameter
 * @param __VA_ARGS__  default value of the parameter
 *
 * @see GKO_ENABLE_LIN_OP_FACTORY for more details, and usage example
 *
 * @ingroup LinOp
 */
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    mutable _name{__VA_ARGS__};                                              \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args &&... _value)                                     \
        const->const ::gko::xstd::decay_t<decltype(*this)> &                 \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Args>(_value)...};                   \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#else  // defined(__CUDACC__) || defined(__HIPCC__)
// A workaround for the NVCC compiler - parameter pack expansion does not work
// properly. You won't be able to use factories in code compiled with NVCC, but
// at least this won't trigger a compiler error as soon as a header using it is
// included. To not get a linker error, we provide a dummy body.
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    mutable _name{__VA_ARGS__};                                              \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args &&... _value)                                     \
        const->const ::gko::xstd::decay_t<decltype(*this)> &                 \
    {                                                                        \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif  // defined(__CUDACC__) || defined(__HIPCC__)


}  // namespace gko


#endif  // GKO_CORE_BASE_LIN_OP_HPP_
