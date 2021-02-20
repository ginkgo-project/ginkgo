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
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/matrix_assembly_data.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class BatchDiagonal;


}


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
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply(const BatchLinOp *, BatchLinOp *)
     */
    const BatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x) const
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
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
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * @copydoc apply(const BatchLinOp *, const BatchLinOp *, const BatchLinOp
     * *, BatchLinOp *)
     */
    const BatchLinOp *apply(const BatchLinOp *alpha, const BatchLinOp *b,
                            const BatchLinOp *beta, BatchLinOp *x) const
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return this;
    }

    /**
     * Returns the size of the operator.
     *
     * @return size of the operator
     */
    const std::vector<dim<2>> &get_sizes() const noexcept { return sizes_; }

    /**
     * Returns the number of batches stored in the operator.
     *
     * @return  number of batches of the operator
     */
    size_type get_num_batches() const noexcept { return num_batches_; }

protected:
    /**
     * Creates a linear operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param size  the size of the operator
     */
    explicit BatchLinOp(
        std::shared_ptr<const Executor> exec,
        const std::vector<dim<2>> &sizes = std::vector<dim<2>>{})
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec),
          sizes_{sizes},
          num_batches_{sizes_.size()}
    {}

    /**
     * Sets the number of batches of the operator.
     *
     * @param value  the new number of batches of the operator
     */
    void set_num_batches(const size_type value) noexcept
    {
        num_batches_ = value;
    }

    /**
     * Sets the size of the operator.
     *
     * @param value  the new size of the operator
     */
    void set_sizes(const std::vector<dim<2>> &value) noexcept
    {
        sizes_ = value;
    }

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
    void validate_application_parameters(const BatchLinOp *b,
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
    void validate_application_parameters(const BatchLinOp *alpha,
                                         const BatchLinOp *b,
                                         const BatchLinOp *beta,
                                         const BatchLinOp *x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            alpha, std::vector<dim<2>>(num_batches_, dim<2>(1, 1)));
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            beta, std::vector<dim<2>>(num_batches_, dim<2>(1, 1)));
    }

private:
    std::vector<dim<2>> sizes_{};
    size_type num_batches_{};
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
        auto generated = AbstractFactory::generate(input);
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
 * Linear operators which support permutation should implement the
 * Permutable interface.
 *
 * It provides functions to permute the rows and columns of a LinOp,
 * independently or symmetrically, and with a regular or inverted permutation.
 *
 * After a regular row permutation with permutation array `perm` the row `i` in
 * the output LinOp contains the row `perm[i]` from the input LinOp.
 * After an inverse row permutation, the row `perm[i]` in the output LinOp
 * contains the row `i` from the input LinOp.
 * Equivalently, after a column permutation, the output stores in column `i`
 * the column `perm[i]` from the input, and an inverse column permutation
 * stores in column `perm[i]` the column `i` from the input.
 * A symmetric permutation is functionally equivalent to calling
 * `as<Permutable>(A->row_permute(perm))->column_permute(perm)`, but the
 * implementation can provide better performance due to kernel fusion.
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
class BatchPermutable {
public:
    virtual ~BatchPermutable() = default;

    /**
     * Returns a BatchLinOp representing the symmetric row and column
     * permutation of the Permutable object. In the resulting LinOp, the entry
     * at location `(i,j)` contains the input value `(perm[i],perm[j])`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<BatchLinOp> permute(
        std::vector<const Array<IndexType> *> permutation_indices) const
    {
        return as<BatchPermutable>(this->row_permute(permutation_indices))
            ->column_permute(permutation_indices);
    };

    /**
     * Returns a BatchLinOp representing the symmetric inverse row and column
     * permutation of the Permutable object.
     * In the resulting LinOp, the entry at location `(perm[i],perm[j])`
     * contains the input value `(i,j)`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<BatchLinOp> inverse_permute(
        std::vector<const Array<IndexType> *> permutation_indices) const
    {
        return as<BatchPermutable>(
                   this->inverse_row_permute(permutation_indices))
            ->inverse_column_permute(permutation_indices);
    };

    /**
     * Returns a BatchLinOp representing the row permutation of the Permutable
     * object.
     * In the resulting BatchLinOp, the row `i` contains the input row
     * `perm[i]`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order.
     *
     * @return a pointer to the new permuted object
     */
    virtual std::unique_ptr<BatchLinOp> row_permute(
        std::vector<const Array<IndexType> *> permutation_indices) const = 0;

    /**
     * Returns a BatchLinOp representing the column permutation of the
     * Permutable object. In the resulting BatchLinOp, the column `i` contains
     * the input column `perm[i]`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new column permuted object
     */
    virtual std::unique_ptr<LinOp> column_permute(
        std::vector<const Array<IndexType> *> permutation_indices) const = 0;

    /**
     * Returns a BatchLinOp representing the row permutation of the inverse
     * permuted object. In the resulting BatchLinOp, the row `perm[i]` contains
     * the input row `i`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<BatchLinOp> inverse_row_permute(
        std::vector<const Array<IndexType> *> permutation_indices) const = 0;

    /**
     * Returns a BatchLinOp representing the row permutation of the inverse
     * permuted object. In the resulting BatchLinOp, the column `perm[i]`
     * contains the input column `i`.
     *
     * @param permutation_indices  the array of indices containing the
     *                             permutation order `perm`.
     *
     * @return a pointer to the new inverse permuted object
     */
    virtual std::unique_ptr<BatchLinOp> inverse_column_permute(
        std::vector<const Array<IndexType> *> permutation_indices) const = 0;
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


/**
 * A LinOp implementing this interface can be preconditioned.
 *
 * @ingroup precond
 * @ingroup BatchLinOp
 */
class BatchPreconditionable {
public:
    virtual ~BatchPreconditionable() = default;

    /**
     * Returns the preconditioner operator used by the Preconditionable.
     *
     * @return the preconditioner operator used by the Preconditionable
     */
    virtual std::shared_ptr<const BatchLinOp> get_preconditioner() const
    {
        return preconditioner_;
    }

    /**
     * Sets the preconditioner operator used by the Preconditionable.
     *
     * @param new_precond  the new preconditioner operator used by the
     *                     Preconditionable
     */
    virtual void set_preconditioner(
        std::shared_ptr<const BatchLinOp> new_precond)
    {
        preconditioner_ = new_precond;
    }

private:
    std::shared_ptr<const BatchLinOp> preconditioner_{};
};


/**
 * The diagonal of a LinOp implementing this interface can be extracted.
 * extract_diagonal extracts the elements whose col and row index are the
 * same and stores the result in a min(nrows, ncols) x 1 dense matrix.
 *
 * @ingroup BatchLinOp
 */
template <typename ValueType>
class BatchDiagonalExtractable {
public:
    using value_type = ValueType;

    virtual ~BatchDiagonalExtractable() = default;

    /**
     * Extracts the diagonal entries of the matrix into a vector.
     *
     * @param diag  the vector into which the diagonal will be written
     */
    virtual std::unique_ptr<matrix::BatchDiagonal<ValueType>> extract_diagonal()
        const = 0;
};


/**
 * The AbsoluteComputable is an interface that allows to get the component wise
 * absolute of a LinOp. Use EnableAbsoluteComputation<AbsoluteLinOp> to
 * implement this interface.
 */
class BatchAbsoluteComputable {
public:
    /**
     * Gets the absolute LinOp
     *
     * @return a pointer to the new absolute LinOp
     */
    virtual std::unique_ptr<BatchLinOp> compute_absolute_linop() const = 0;

    /**
     * Compute absolute inplace on each element.
     */
    virtual void compute_absolute_inplace() = 0;
};


/**
 * The EnableAbsoluteComputation mixin provides the default implementations of
 * `compute_absolute_linop` and the absolute interface. `compute_absolute` gets
 * a new AbsoluteLinOp. `compute_absolute_inplace` applies absolute
 * inplace, so it still keeps the value_type of the class.
 *
 * @tparam AbsoluteLinOp  the absolute LinOp which is being returned
 *                        [CRTP parameter]
 *
 * @ingroup BatchLinOp
 */
template <typename AbsoluteBatchLinOp>
class EnableBatchAbsoluteComputation : public BatchAbsoluteComputable {
public:
    using absolute_type = AbsoluteBatchLinOp;

    virtual ~EnableBatchAbsoluteComputation() = default;

    std::unique_ptr<BatchLinOp> compute_absolute_linop() const override
    {
        return this->compute_absolute();
    }

    /**
     * Gets the AbsoluteLinOp
     *
     * @return a pointer to the new absolute object
     */
    virtual std::unique_ptr<absolute_type> compute_absolute() const = 0;
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
 *     or GKO_ENABLE_BATCH_LIN_OP_FACTORY macro (used for operators created from
 * other operators, like preconditioners and solvers).
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
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteBatchLinOp *apply(const BatchLinOp *b, BatchLinOp *x)
    {
        this->validate_application_parameters(b, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    const ConcreteBatchLinOp *apply(const BatchLinOp *alpha,
                                    const BatchLinOp *b, const BatchLinOp *beta,
                                    BatchLinOp *x) const
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

    ConcreteBatchLinOp *apply(const BatchLinOp *alpha, const BatchLinOp *b,
                              const BatchLinOp *beta, BatchLinOp *x)
    {
        this->validate_application_parameters(alpha, b, beta, x);
        auto exec = this->get_executor();
        this->apply_impl(make_temporary_clone(exec, alpha).get(),
                         make_temporary_clone(exec, b).get(),
                         make_temporary_clone(exec, beta).get(),
                         make_temporary_clone(exec, x).get());
        return self();
    }

protected:
    GKO_ENABLE_SELF(ConcreteBatchLinOp);
};


/**
 * This is an alias for the EnableDefaultFactory mixin, which correctly sets the
 * template parameters to enable a subclass of BatchLinOpFactory.
 *
 * @tparam ConcreteFactory  the concrete factory which is being implemented
 *                          [CRTP parmeter]
 * @tparam ConcreteBatchLinOp  the concrete BatchLinOp type which this factory
 * produces, needs to have a constructor which takes a const ConcreteFactory *,
 * and an std::shared_ptr<const BatchLinOp> as parameters.
 * @tparam ParametersType  a subclass of enable_parameters_type template which
 *                         defines all of the parameters of the factory
 * @tparam PolymorphicBase  parent of ConcreteFactory in the polymorphic
 *                          hierarchy, has to be a subclass of BatchLinOpFactory
 *
 * @ingroup BatchLinOp
 */
template <typename ConcreteFactory, typename ConcreteBatchLinOp,
          typename ParametersType, typename PolymorphicBase = BatchLinOpFactory>
using EnableDefaultBatchLinOpFactory =
    EnableDefaultFactory<ConcreteFactory, ConcreteBatchLinOp, ParametersType,
                         PolymorphicBase>;

/**
 * This Macro will generate a new type containing the parameters for the factory
 * `_factory_name`. For more details, see #GKO_ENABLE_BATCH_LIN_OP_FACTORY().
 * It is required to use this macro **before** calling the
 * macro #GKO_ENABLE_BATCH_LIN_OP_FACTORY().
 * It is also required to use the same names for all parameters between both
 * macros.
 *
 * @param _parameters_name  name of the parameters member in the class
 * @param _factory_name  name of the generated factory type
 *
 * @ingroup BatchLinOp
 */
#define GKO_CREATE_FACTORY_PARAMETERS(_parameters_name, _factory_name)  \
public:                                                                 \
    class _factory_name;                                                \
    struct _parameters_name##_type                                      \
        : public ::gko::enable_parameters_type<_parameters_name##_type, \
                                               _factory_name>


/**
 * This macro will generate a default implementation of a BatchLinOpFactory for
 * the BatchLinOp subclass it is defined in.
 *
 * It is required to first call the macro #GKO_CREATE_FACTORY_PARAMETERS()
 * before this one in order to instantiate the parameters type first.
 *
 * The list of parameters for the factory should be defined in a code block
 * after the macro definition, and should contain a list of
 * GKO_FACTORY_PARAMETER_* declarations. The class should provide a constructor
 * with signature
 * _lin_op(const _factory_name *, std::shared_ptr<const BatchLinOp>)
 * which the factory will use a callback to construct the object.
 *
 * A minimal example of a linear operator is the following:
 *
 * ```c++
 * struct MyBatchLinOp : public EnableBatchLinOp<MyBatchLinOp> {
 *     GKO_ENABLE_BATCH_LIN_OP_FACTORY(MyBatchLinOp, my_parameters, Factory) {
 *         // a factory parameter named "my_value", of type int and default
 *         // value of 5
 *         int GKO_FACTORY_PARAMETER_SCALAR(my_value, 5);
 *         // a factory parameter named `my_pair` of type `std::pair<int,int>`
 *         // and default value {5, 5}
 *         std::pair<int, int> GKO_FACTORY_PARAMETER_VECTOR(my_pair, 5, 5);
 *     };
 *     // constructor needed by EnableBatchLinOp
 *     explicit MyBatchLinOp(std::shared_ptr<const Executor> exec) {
 *         : EnableBatchLinOp<MyBatchLinOp>(exec) {}
 *     // constructor needed by the factory
 *     explicit MyBatchLinOp(const Factory *factory,
 *                      std::shared_ptr<const BatchLinOp> matrix)
 *         : EnableBatchLinOp<MyBatchLinOp>(factory->get_executor()),
 * matrix->get_size()),
 *           // store factory's parameters locally
 *           my_parameters_{factory->get_parameters()},
 *     {
 *          int value = my_parameters_.my_value;
 *          // do something with value
 *     }
 * ```
 *
 * MyBatchLinOp can then be created as follows:
 *
 * ```c++
 * auto exec = gko::ReferenceExecutor::create();
 * // create a factory with default `my_value` parameter
 * auto fact = MyBatchLinOp::build().on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 5
 *
 * // create a factory with custom `my_value` parameter
 * auto fact = MyBatchLinOp::build().with_my_value(0).on(exec);
 * // create a operator using the factory:
 * auto my_op = fact->generate(gko::matrix::Identity::create(exec, 2));
 * std::cout << my_op->get_my_parameters().my_value;  // prints 0
 * ```
 *
 * @note It is possible to combine both the #GKO_CREATE_FACTORY_PARAMETER_*()
 * macros with this one in a unique macro for class __templates__ (not with
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
 * @ingroup BatchLinOp
 */
#define GKO_ENABLE_BATCH_LIN_OP_FACTORY(_lin_op, _parameters_name,             \
                                        _factory_name)                         \
public:                                                                        \
    const _parameters_name##_type &get_##_parameters_name() const              \
    {                                                                          \
        return _parameters_name##_;                                            \
    }                                                                          \
                                                                               \
    class _factory_name                                                        \
        : public ::gko::EnableDefaultBatchLinOpFactory<                        \
              _factory_name, _lin_op, _parameters_name##_type> {               \
        friend class ::gko::EnablePolymorphicObject<_factory_name,             \
                                                    ::gko::BatchLinOpFactory>; \
        friend class ::gko::enable_parameters_type<_parameters_name##_type,    \
                                                   _factory_name>;             \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec)    \
            : ::gko::EnableDefaultBatchLinOpFactory<_factory_name, _lin_op,    \
                                                    _parameters_name##_type>(  \
                  std::move(exec))                                             \
        {}                                                                     \
        explicit _factory_name(std::shared_ptr<const ::gko::Executor> exec,    \
                               const _parameters_name##_type &parameters)      \
            : ::gko::EnableDefaultBatchLinOpFactory<_factory_name, _lin_op,    \
                                                    _parameters_name##_type>(  \
                  std::move(exec), parameters)                                 \
        {}                                                                     \
    };                                                                         \
    friend ::gko::EnableDefaultBatchLinOpFactory<_factory_name, _lin_op,       \
                                                 _parameters_name##_type>;     \
                                                                               \
                                                                               \
private:                                                                       \
    _parameters_name##_type _parameters_name##_;                               \
                                                                               \
public:                                                                        \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


/**
 * Defines a build method for the factory, simplifying its construction by
 * removing the repetitive typing of factory's name.
 *
 * @param _factory_name  the factory for which to define the method
 *
 * @ingroup BatchLinOp
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
 * @see GKO_ENABLE_BATCH_LIN_OP_FACTORY for more details, and usage example
 *
 * @deprecated Use GKO_FACTORY_PARAMETER_SCALAR or GKO_FACTORY_PARAMETER_VECTOR
 *
 * @ingroup BatchLinOp
 */
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    mutable _name{__VA_ARGS__};                                              \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args &&... _value)                                     \
        const->const std::decay_t<decltype(*this)> &                         \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Args>(_value)...};                   \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

/**
 * Creates a scalar factory parameter in the factory parameters structure.
 *
 * Scalar in this context means that the constructor for this type only takes
 * a single parameter.
 *
 * @param _name  name of the parameter
 * @param _default  default value of the parameter
 *
 * @see GKO_ENABLE_BATCH_LIN_OP_FACTORY for more details, and usage example
 *
 * @ingroup BatchLinOp
 */
#define GKO_FACTORY_PARAMETER_SCALAR(_name, _default) \
    GKO_FACTORY_PARAMETER(_name, _default)

/**
 * Creates a vector factory parameter in the factory parameters structure.
 *
 * Vector in this context means that the constructor for this type takes
 * multiple parameters.
 *
 * @param _name  name of the parameter
 * @param _default  default value of the parameter
 *
 * @see GKO_ENABLE_BATCH_LIN_OP_FACTORY for more details, and usage example
 *
 * @ingroup BatchLinOp
 */
#define GKO_FACTORY_PARAMETER_VECTOR(_name, ...) \
    GKO_FACTORY_PARAMETER(_name, __VA_ARGS__)
#else  // defined(__CUDACC__) || defined(__HIPCC__)
// A workaround for the NVCC compiler - parameter pack expansion does not work
// properly, because while the assignment to a scalar value is translated by
// cudafe into a C-style cast, the parameter pack expansion is not removed and
// `Args&&... args` is still kept as a parameter pack.
#define GKO_FACTORY_PARAMETER(_name, ...)                                    \
    mutable _name{__VA_ARGS__};                                              \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args &&... _value)                                     \
        const->const std::decay_t<decltype(*this)> &                         \
    {                                                                        \
        GKO_NOT_IMPLEMENTED;                                                 \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_FACTORY_PARAMETER_SCALAR(_name, _default)                        \
    mutable _name{_default};                                                 \
                                                                             \
    template <typename Arg>                                                  \
    auto with_##_name(Arg &&_value)                                          \
        const->const std::decay_t<decltype(*this)> &                         \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Arg>(_value)};                       \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")

#define GKO_FACTORY_PARAMETER_VECTOR(_name, ...)                             \
    mutable _name{__VA_ARGS__};                                              \
                                                                             \
    template <typename... Args>                                              \
    auto with_##_name(Args &&... _value)                                     \
        const->const std::decay_t<decltype(*this)> &                         \
    {                                                                        \
        using type = decltype(this->_name);                                  \
        this->_name = type{std::forward<Args>(_value)...};                   \
        return *this;                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif  // defined(__CUDACC__) || defined(__HIPCC__)


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
