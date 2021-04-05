/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_


#include <memory>
#include <type_traits>
#include <utility>


#include <ginkgo/core/base/abstract_factory.hpp>
#include <ginkgo/core/base/batch_lin_op_helpers.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/log/logger.hpp>


namespace gko {


/**
 * @addtogroup BatchLinOp
 *
 * @section linop_concept Linear operator as a concept
 *
 * The linear operator (BatchLinOp) is a base class for all linear algebra
 * objects in Ginkgo. The main benefit of having a single base class for the
 * entire collection of linear algebra objects (as opposed to having separate
 * hierarchies for matrices, solvers and preconditioners) is the generality
 * it provides.
 *
 * First, since all subclasses provide a common interface, the library users are
 * exposed to a smaller set of routines. For example, a
 * matrix-vector product, a preconditioner application, or even a system solve
 * are just different terms given to the operation of applying a certain linear
 * operator to a vector. As such, Ginkgo uses the same routine name,
 * BatchLinOp::apply() for each of these operations, where the actual
 * operation performed depends on the type of linear operator involved in
 * the operation.
 *
 * Second, a common interface often allows for writing more generic code. If a
 * user's routine requires only operations provided by the BatchLinOp interface,
 * the same code can be used for any kind of linear operators, independent of
 * whether these are matrices, solvers or preconditioners. This feature is also
 * extensively used in Ginkgo itself. For example, a preconditioner used
 * inside a Krylov solver is a BatchLinOp. This allows the user to supply a wide
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
 * Finally, direct manipulation of BatchLinOp objects is rarely required in
 * simple scenarios. As an illustrative example, one could construct a
 * fixed-point iteration routine $x_{k+1} = Lx_k + b$ as follows:
 *
 * ```cpp
 * std::unique_ptr<matrix::Dense<>> calculate_fixed_point(
 *         int iters, const BatchLinOp *L, const matrix::Dense<> *x0
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
 * Here, if $L$ is a matrix, BatchLinOp::apply() refers to the matrix vector
 * product, and `L->apply(a, b)` computes $b = L \cdot a$.
 * `x->add_scaled(one.get(), b.get())` is the `axpy` vector update $x:=x+b$.
 *
 * The interesting part of this example is the apply() routine at line 4 of the
 * function body. Since this routine is part of the BatchLinOp base class, the
 * fixed-point iteration routine can calculate a fixed point not only for
 * matrices, but for any type of linear operator.
 *
 * @ref BatchLinOp
 */
class BatchLinOp : public EnableAbstractPolymorphicObject<BatchLinOp> {
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
    BatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x)
    {
        this->template log<log::Logger::batch_linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_apply_completed>(this, b,
                                                                     x);
        return this;
    }

    /**
     * @copydoc apply(const BatchLinOp *, BatchLinOp *)
     */
    const BatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x) const
    {
        this->template log<log::Logger::batch_linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_apply_completed>(this, b,
                                                                     x);
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
    BatchLinOp *apply(const BatchLinOp *alpha, const BatchLinOp *b,
                      const BatchLinOp *beta, BatchLinOp *x)
    {
        this->template log<log::Logger::batch_linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return this;
    }

    /**
     * @copydoc apply(const BatchLinOp *, const BatchLinOp *, const BatchLinOp
     * *, BatchLinOp *)
     */
    const BatchLinOp *apply(const BatchLinOp *alpha, const BatchLinOp *b,
                            const BatchLinOp *beta, BatchLinOp *x) const
    {
        this->template log<log::Logger::batch_linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return this;
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    size_type get_num_batches() const noexcept
    {
        return size_.get_num_batches();
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    void set_size(const batch_dim &size) { size_ = size; }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    const batch_dim &get_size() const noexcept { return size_; }

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
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const size_type num_batches = 0,
                        const dim<2> &size = dim<2>{})
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec),
          size_{num_batches > 0 ? batch_dim(num_batches, size) : batch_dim{}}
    {}

    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const std::vector<dim<2>> &batch_sizes)
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec),
          size_{batch_dim(batch_sizes)}
    {}

    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const batch_dim &batch_sizes)
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec), size_{batch_sizes}
    {}

    /**
     * Implementers of BatchLinOp should override this function instead
     * of apply(const BatchLinOp *, BatchLinOp *).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input vector(s) on which the operator is applied
     * @param x  the output vector(s) where the result is stored
     */
    virtual void apply_impl(const BatchLinOp *b, BatchLinOp *x) const = 0;

    /**
     * Implementers of BatchLinOp should override this function instead
     * of apply(const BatchLinOp *, const BatchLinOp *, const BatchLinOp *,
     * BatchLinOp *).
     *
     * @param alpha  scaling of the result of op(b)
     * @param b  vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output vector(s)
     */
    virtual void apply_impl(const BatchLinOp *alpha, const BatchLinOp *b,
                            const BatchLinOp *beta, BatchLinOp *x) const = 0;

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param b  vector(s) on which the operator is applied
     * @param x  output vector(s)
     */
    virtual void validate_application_parameters(const BatchLinOp *b,
                                                 const BatchLinOp *x) const
    {
        GKO_ASSERT_BATCH_CONFORMANT(this, b);
        GKO_ASSERT_BATCH_EQUAL_ROWS(this, x);
        GKO_ASSERT_BATCH_EQUAL_COLS(b, x);
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
    virtual void validate_application_parameters(const BatchLinOp *alpha,
                                                 const BatchLinOp *b,
                                                 const BatchLinOp *beta,
                                                 const BatchLinOp *x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            alpha, batch_dim(b->get_num_batches(), dim<2>(1, 1)));
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            beta, batch_dim(b->get_num_batches(), dim<2>(1, 1)));
    }

private:
    batch_dim size_{};
};


/**
 * A BatchLinOpFactory represents a higher order mapping which transforms one
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
 * Another example of a BatchLinOpFactory is a preconditioner. A preconditioner
 * for a linear operator $A$ is a linear operator $M^{-1}$, which approximates
 * $A^{-1}$. In addition, it is stored in a way such that both the data of
 * $M^{-1}$ is cheap to compute from $A$, and the operation $x = M^{-1}b$ can be
 * computed quickly. These operators are useful to accelerate the convergence of
 * Krylov solvers. Thus, a preconditioner also fits into the BatchLinOpFactory
 * framework:
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
 * @ingroup BatchLinOp
 */
class BatchLinOpFactory
    : public AbstractFactory<BatchLinOp, std::shared_ptr<const BatchLinOp>> {
public:
    using AbstractFactory<BatchLinOp,
                          std::shared_ptr<const BatchLinOp>>::AbstractFactory;

    std::unique_ptr<BatchLinOp> generate(
        std::shared_ptr<const BatchLinOp> input) const
    {
        this->template log<log::Logger::batch_linop_factory_generate_started>(
            this, input.get());
        auto generated = AbstractFactory::generate(input);
        this->template log<log::Logger::batch_linop_factory_generate_completed>(
            this, input.get(), generated.get());
        return generated;
    }
};


/**
 * The EnableBatchLinOp mixin can be used to provide sensible default
 * implementations of the majority of the BatchLinOp and PolymorphicObject
 * interface.
 *
 * The goal of the mixin is to facilitate the development of new BatchLinOp, by
 * enabling the implementers to focus on the important parts of their operator,
 * while the library takes care of generating the trivial utility functions.
 * The mixin will provide default implementations for the entire
 * PolymorphicObject interface, including a default implementation of
 * `copy_from` between objects of the new BatchLinOp type. It will also hide the
 * default BatchLinOp::apply() methods with versions that preserve the static
 * type of the object.
 *
 * Implementers of new BatchLinOps are required to specify only the following
 * aspects:
 *
 * 1.  Creation of the BatchLinOp: This can be facilitated via either
 *     EnableCreateMethod mixin (used mostly for matrix formats),
 *     or GKO_ENABLE_LIN_OP_FACTORY macro (used for operators created from other
 *     operators, like preconditioners and solvers).
 * 2.  Application of the BatchLinOp: Implementers have to override the two
 *     overloads of the BatchLinOp::apply_impl() virtual methods.
 *
 * @tparam ConcreteBatchLinOp  the concrete BatchLinOp which is being
 * implemented [CRTP parameter]
 * @tparam PolymorphicBase  parent of ConcreteBatchLinOp in the polymorphic
 *                          hierarchy, has to be a subclass of BatchLinOp
 *
 * @ingroup BatchLinOp
 */
template <typename ConcreteBatchLinOp, typename PolymorphicBase = BatchLinOp>
class EnableBatchLinOp
    : public EnablePolymorphicObject<ConcreteBatchLinOp, PolymorphicBase>,
      public EnablePolymorphicAssignment<ConcreteBatchLinOp> {
public:
    using EnablePolymorphicObject<ConcreteBatchLinOp,
                                  PolymorphicBase>::EnablePolymorphicObject;

    const ConcreteBatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x) const
    {
        this->template log<log::Logger::batch_linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_apply_completed>(this, b,
                                                                     x);
        return self();
    }

    ConcreteBatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x)
    {
        this->template log<log::Logger::batch_linop_apply_started>(this, b, x);
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_apply_completed>(this, b,
                                                                     x);
        return self();
    }

    const ConcreteBatchLinOp *apply(const BatchLinOp *alpha,
                                    const BatchLinOp *b, const BatchLinOp *beta,
                                    BatchLinOp *x) const
    {
        this->template log<log::Logger::batch_linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return self();
    }

    ConcreteBatchLinOp *apply(const BatchLinOp *alpha, const BatchLinOp *b,
                              const BatchLinOp *beta, BatchLinOp *x)
    {
        this->template log<log::Logger::batch_linop_advanced_apply_started>(
            this, alpha, b, beta, x);
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        this->template log<log::Logger::batch_linop_advanced_apply_completed>(
            this, alpha, b, beta, x);
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteBatchLinOp);
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
class BatchTransposable {
public:
    virtual ~BatchTransposable() = default;

    /**
     * Returns a LinOp representing the transpose of the Transposable object.
     *
     * @return a pointer to the new transposed object
     */
    virtual std::unique_ptr<BatchLinOp> transpose() const = 0;

    /**
     * Returns a LinOp representing the conjugate transpose of the Transposable
     * object.
     *
     * @return a pointer to the new conjugate transposed object
     */
    virtual std::unique_ptr<BatchLinOp> conj_transpose() const = 0;
};


/**
 * A BatchLinOp implementing this interface can read its data from a matrix_data
 * structure.
 *
 * @ingroup BatchLinOp
 */
template <typename ValueType, typename IndexType>
class BatchReadableFromMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~BatchReadableFromMatrixData() = default;

    /**
     * Reads a matrix from a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void read(
        const std::vector<matrix_data<ValueType, IndexType>> &data) = 0;

    /**
     * Reads a matrix from a matrix_assembly_data structure.
     *
     * @param data  the matrix_assembly_data structure
     */
    void read(const std::vector<matrix_assembly_data<ValueType, IndexType>>
                  &assembly_data)
    {
        auto mat_data = std::vector<matrix_data<ValueType, IndexType>>(
            assembly_data.size());
        size_type ind = 0;
        for (const auto &i : assembly_data) {
            mat_data[ind] = i.get_ordered_data();
            ++ind;
        }
        this->read(mat_data);
    }
};


/**
 * A BatchLinOp implementing this interface can write its data to a matrix_data
 * structure.
 *
 * @ingroup BatchLinOp
 */
template <typename ValueType, typename IndexType>
class BatchWritableToMatrixData {
public:
    using value_type = ValueType;
    using index_type = IndexType;

    virtual ~BatchWritableToMatrixData() = default;

    /**
     * Writes a matrix to a matrix_data structure.
     *
     * @param data  the matrix_data structure
     */
    virtual void write(
        std::vector<matrix_data<ValueType, IndexType>> &data) const = 0;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
