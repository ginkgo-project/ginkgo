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
 * @section batch_linop_concept Batched Linear operator as a concept
 *
 * A batch linear operator (BatchLinOp) forms the base class for all batched
 * linear algebra objects. In general, it follows the same structure as the
 * LinOp class, but has some crucial differences which make it not strictly
 * representable through or with the LinOp class.
 *
 * A batched operator is defined as a set of independent linear operators which
 * have no communication/information exchange between them. Therefore, any
 * collective operations between the batches is not possible and not
 * implemented. This allows for each batch to be computed and operated on in an
 * embarrasingly parallel fashion.
 *
 * Similar to the LinOp class, the BatchLinOp also implements
 * BatchLinOp::apply() methods which call the internal apply_impl() methods
 * which the concrete BatchLinOp's have to implement.
 *
 * A key difference between the LinOp and the BatchLinOp classes is the storing
 * of dimensions. BatchLinOp allows for storing non-equal objects in the
 * batches and hence stores a batch_dim object instead of a dim object. The
 * batch_dim object is optimized to store less amount of data when storing
 * uniform batches.
 *
 * All size validation functions again verify first that the number of batches
 * are conformant and that the dimensions in the corresponding batches
 * themselves are also valid/conformant. Here too, optimizations for uniform
 * batches have been added.
 *
 * @ref BatchLinOp
 */
class BatchLinOp : public EnableAbstractPolymorphicObject<BatchLinOp> {
public:
    /**
     * Applies a batch linear operator to a batch vector (or a sequence of batch
     * of vectors).
     *
     * Performs the operation x = op(b), where op is this batch linear operator.
     *
     * @param b  the input batch vector(s) on which the batch operator is
     *           applied
     * @param x  the output batch vector(s) where the result is stored
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
     * Returns the number of batches in the batch operator.
     *
     * @return number of batches in the batch operator
     */
    size_type get_num_batch_entries() const noexcept
    {
        return size_.get_num_batch_entries();
    }

    /**
     * Sets the size of the batch operator.
     *
     * @param size to be set
     */
    void set_size(const batch_dim<2> &size) { size_ = size; }

    /**
     * Returns the size of the batch operator.
     *
     * @return size of the batch operator
     */
    const batch_dim<2> &get_size() const noexcept { return size_; }

    /**
     * Returns true if the batch operator uses the data given in x as
     * an initial guess. Returns false otherwise.
     *
     * @return true if the batch operator uses the data given in x as
     *         an initial guess. Returns false otherwise.
     */
    virtual bool apply_uses_initial_guess() const { return false; }

protected:
    /**
     * Creates a batch operator with uniform batches.
     *
     * @param exec        the executor where all the operations are performed
     * @param num_batch_entries the number of batches to be stored in the
     * operator
     * @param size        the size of on of the operator in the batched operator
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const size_type num_batch_entries = 0,
                        const dim<2> &size = dim<2>{})
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec),
          size_{num_batch_entries > 0 ? batch_dim<2>(num_batch_entries, size)
                                      : batch_dim<2>{}}
    {}

    /**
     * Creates a batch operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param batch_sizes  the vector containing the sizes of the batches to be
     * stored in the batch operator.
     *
     * @note If possible and you have uniform batches, please prefer to use the
     * constructor above, as it optimizes the size validations and hence can be
     * significantly faster
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const std::vector<dim<2>> &batch_sizes)
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec),
          size_{batch_dim<2>(batch_sizes)}
    {}

    /**
     * Creates a batch operator.
     *
     * @param exec  the executor where all the operations are performed
     * @param batch_sizes  the sizes of the batch operator stored as a batch_dim
     */
    explicit BatchLinOp(std::shared_ptr<const Executor> exec,
                        const batch_dim<2> &batch_sizes)
        : EnableAbstractPolymorphicObject<BatchLinOp>(exec), size_{batch_sizes}
    {}

    /**
     * Implementers of BatchLinOp should override this function instead
     * of apply(const BatchLinOp *, BatchLinOp *).
     *
     * Performs the operation x = op(b), where op is this linear operator.
     *
     * @param b  the input batch vector(s) on which the operator is applied
     * @param x  the output batch vector(s) where the result is stored
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
     * @param b  batch vector(s) on which the operator is applied
     * @param x  output batch vector(s)
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
     * @param b  batch vector(s) on which the operator is applied
     * @param beta  scaling of the input x
     * @param x  output batch vector(s)
     */
    void validate_application_parameters(const BatchLinOp *alpha,
                                         const BatchLinOp *b,
                                         const BatchLinOp *beta,
                                         const BatchLinOp *x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            alpha, batch_dim<2>(b->get_num_batch_entries(), dim<2>(1, 1)));
        GKO_ASSERT_BATCH_EQUAL_DIMENSIONS(
            beta, batch_dim<2>(b->get_num_batch_entries(), dim<2>(1, 1)));
    }

private:
    batch_dim<2> size_{};
};


/**
 * A BatchLinOpFactory represents a higher order mapping which transforms one
 * batch linear operator into another.
 *
 * In a similar fashion to LinOps, BatchLinOps are also "generated" from the
 * BatchLinOpFactory. A function of this class is to provide a generate method,
 * which internally cals the generate_impl(), which the concrete BatchLinOps
 * have to implement.
 *
 * Example: using BatchCG in Ginkgo
 * ---------------------------
 *
 * ```c++
 * // Suppose A is a batch matrix, batch_b a batch rhs vector, and batch_x an
 * // initial guess
 * // Create a BatchCG which runs for at most 1000 iterations, and stops after
 * // reducing the residual norm by 6 orders of magnitude
 * auto batch_cg_factory = solver::BatchCg<>::build()
 *     .with_max_iters(1000)
 *     .with_rel_residual_goal(1e-6)
 *     .on(cuda);
 * // create a batch linear operator which represents the solver
 * auto batch_cg = batch_cg_factory->generate(A);
 * // solve the system
 * batch_cg->apply(gko::lend(batch_b), gko::lend(batch_x));
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
 *     or GKO_ENABLE_BATCH_LIN_OP_FACTORY macro (used for operators created from
 *     other operators, like preconditioners and solvers).
 * 2.  Application of the BatchLinOp: Implementers have to override the two
 *     overloads of the BatchLinOp::apply_impl() virtual methods.
 *
 * @tparam ConcreteBatchLinOp  the concrete BatchLinOp which is being
 *                             implemented [CRTP parameter]
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
 * Batch Linear operators which support transposition of the distinct batch
 * entries should implement the BatchTransposable interface.
 *
 * It provides two functionalities, the normal transpose and the
 * conjugate transpose, both transposing the invidual batch entries.
 *
 * The normal transpose returns the transpose of the linear operator without
 * changing any of its elements representing the operation, $B = A^{T}$.
 *
 * The conjugate transpose returns the conjugate of each of the elements and
 * additionally transposes the linear operator representing the operation, $B
 * = A^{H}$.
 *
 * Example: Transposing a BatchCsr matrix:
 * ------------------------------------
 *
 * ```c++
 * //Transposing an object of BatchLinOp type.
 * //The object you want to transpose.
 * auto op = matrix::BatchCsr::create(exec);
 * //Transpose the object by first converting it to a transposable type.
 * auto trans = op->transpose();
 * ```
 */
class BatchTransposable {
public:
    virtual ~BatchTransposable() = default;

    /**
     * Returns a BatchLinOp containing the transposes of the distinct entries of
     * the BatchTransposable object.
     *
     * @return a pointer to the new transposed object
     */
    virtual std::unique_ptr<BatchLinOp> transpose() const = 0;

    /**
     * Returns a BatchLinOp containing the conjugate transpose of the distinct
     * entries of the BatchTransposable object.
     *
     * @return a pointer to the new conjugate transposed object
     */
    virtual std::unique_ptr<BatchLinOp> conj_transpose() const = 0;
};


/**
 * A batch made up of entities that can scaled from the left and right by
 * scaling vectors.
 *
 * TODO: Somehow restrict this to be a BatchLinOp. Mixin?
 */
class BatchScalable {
public:
    virtual ~BatchScalable() = default;

    /**
     * Scales each matrix in a batch from the left and right.
     *
     * @param left_scale  The left scaling batch vector.
     *                    In case of matrices, this scales the rows.
     * @param right_scale  The right scaling batch vector.
     *                     In case of matrices, this scales the columns.
     */
    virtual void batch_scale(const BatchLinOp *left_scale,
                             const BatchLinOp *right_scale) = 0;
};


namespace matrix {


template <typename ValueType>
class BatchDense;


}


/**
 * Default batch scalable interface with some type-checking.
 *
 * @see BatchScalable
 */
template <typename ValueType>
class EnableBatchScaling : public BatchScalable {
public:
    /**
     * Scales each matrix in a batch from the left and right.
     *
     * @param left_scale  The left scaling batch vector.
     * @param right_scale  The right scaling batch vector.
     *
     * @throw gko::NotSupported  If the arguments do not point to BatchDense
     * objects.
     */
    void batch_scale(const BatchLinOp *const left_scale,
                     const BatchLinOp *const right_scale) override
    {
        if (!left_scale && !right_scale) {
            return;
        }

        if (left_scale) {
            if (!dynamic_cast<const matrix::BatchDense<ValueType> *>(
                    left_scale))
                GKO_NOT_SUPPORTED(left_scale);
        }
        if (right_scale) {
            if (!dynamic_cast<const matrix::BatchDense<ValueType> *>(
                    right_scale))
                GKO_NOT_SUPPORTED(right_scale);
        }

        /* TODO: Somehow restrict BatchScalable and EnableBatchScaling to
         * BatchLinOp so that the following dynamic_cast is not needed.
         */
        if (auto blp = dynamic_cast<BatchLinOp *>(this)) {
            GKO_ASSERT_BATCH_SCALABLE_TWO_SIDED(blp, left_scale, right_scale);
        } else {
            GKO_NOT_SUPPORTED(this);
        }

        this->batch_scale_impl(left_scale, right_scale);
    }

protected:
    virtual void batch_scale_impl(const BatchLinOp *left_scale,
                                  const BatchLinOp *right_scale) = 0;
};


/**
 * Provides basic functionality for solvers which need to be composed with
 * batch scaling.
 *
 * If A is a matrix, S is a batch-scaling operation and F is a solver-type
 * such that F(A) is a solver,
 * then if the class for F inherits from this EnableBatchScaledSolver class,
 * it represents a new solver F(S(A)) (F composed with S applied to A).
 */
template <typename ValueType>
class EnableBatchScaledSolver : public EnableBatchScaling<ValueType> {
public:
    using value_type = ValueType;

    const matrix::BatchDense<value_type> *get_left_scaling_vector() const
    {
        return left_scale_;
    }

    const matrix::BatchDense<value_type> *get_right_scaling_vector() const
    {
        return right_scale_;
    }

protected:
    /**
     * Sets the batch scaling vectors so that they can accessed
     * by the solver kernels.
     *
     * @post Once batch_scale, and thus this function, are called on a solver,
     * it (its apply function) needs to behave as though the underlying matrix
     * has been scaled appropriately.
     *
     * @param left_scale_op  Left scaling batch Dense vector.
     * @param right_scale_op  Right scaling batch Dense vector.
     */
    void batch_scale_impl(const BatchLinOp *const left_scale_op,
                          const BatchLinOp *const right_scale_op) override
    {
        // auto sysmat =
        // std::static_pointer_cast<ConcreteSolver*>()->system_matrix_; auto
        // orig_const_mat = 	std::dynamic_pointer_cast<const
        // BatchScalable>(sysmat); if (!orig_const_mat) {
        // 	GKO_NOT_SUPPORTED(sysmat);
        // }
        // auto mat = std::const_pointer_cast<BatchScalable>(orig_const_mat);
        // mat->batch_scale(left_op, right_op);
        left_scale_ =
            static_cast<const matrix::BatchDense<ValueType> *>(left_scale_op);
        right_scale_ =
            static_cast<const matrix::BatchDense<ValueType> *>(right_scale_op);
    }

    const matrix::BatchDense<value_type> *left_scale_ = nullptr;
    const matrix::BatchDense<value_type> *right_scale_ = nullptr;
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
     * Reads a batch matrix from a std::vector of matrix_data objects.
     *
     * @param data  the std::vector of matrix_data objects
     */
    virtual void read(
        const std::vector<matrix_data<ValueType, IndexType>> &data) = 0;

    /**
     * Reads a matrix from a std::vector of matrix_assembly_data objects.
     *
     * @param data  the std::vector of matrix_assembly_data objects
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
 * A BatchLinOp implementing this interface can write its data to a std::vector
 * of matrix_data objects.
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
