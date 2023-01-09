/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


class batch_stride {
public:
    /**
     * Checks if the batch_stride object stores equal sizes.
     *
     * @return bool representing whether equal strides are being stored
     */
    bool stores_equal_strides() const { return equal_strides_; }

    /**
     * Get the number of batche entries stored
     *
     * @return num_batch_entries
     */
    size_type get_num_batch_entries() const { return num_batch_entries_; }

    /**
     * Get the batch strides as a std::vector.
     *
     * @return  the std::vector of batch strides
     */
    std::vector<size_type> get_batch_strides() const
    {
        if (!equal_strides_) {
            return strides_;
        } else {
            return std::vector<size_type>(num_batch_entries_, common_stride_);
        }
    }

    /**
     * Get the batch size of a particular entry in the batch.
     *
     * @param batch_entry the entry whose size is needed
     *
     * @return  the size of the batch entry at the requested index
     */
    const size_type& at(const size_type batch_entry = 0) const
    {
        if (equal_strides_) {
            return common_stride_;
        } else {
            GKO_ASSERT(batch_entry < num_batch_entries_);
            return strides_[batch_entry];
        }
    }

    /**
     * Checks if two batch_stride objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend bool operator==(const batch_stride& x, const batch_stride& y)
    {
        if (x.equal_strides_ && y.equal_strides_) {
            return x.num_batch_entries_ == y.num_batch_entries_ &&
                   x.common_stride_ == y.common_stride_;
        } else {
            return x.strides_ == y.strides_;
        }
    }

    /**
     * Creates a batch_stride object which stores uniform batch strides.
     *
     * @param num_batch_entries  number of batche entries to be stored
     * @param common_stride  the common stride of all the batch entries to be
     * stored
     *
     * @note  Use this constructor when uniform batches need to be stored.
     */
    batch_stride(const size_type num_batch_entries = 0,
                 const size_type& common_stride = 0)
        : equal_strides_(true),
          common_stride_(common_stride),
          num_batch_entries_(num_batch_entries),
          strides_()
    {}

    /**
     * Creates a batch_stride object which stores possibly non-uniform batch
     * strides.
     *
     * @param batch_strides  the std::vector object that stores the
     * batch_strides
     *
     * @note  Use this constructor when non-uniform batches need to be stored.
     */
    batch_stride(const std::vector<size_type>& batch_strides)
        : equal_strides_(false),
          common_stride_(size_type{}),
          num_batch_entries_(batch_strides.size()),
          strides_(batch_strides)
    {
        check_equal_strides();
    }

private:
    inline void check_equal_strides()
    {
        for (size_type b = 1; b < num_batch_entries_; ++b) {
            if (strides_[0] != strides_[b]) {
                equal_strides_ = false;
                common_stride_ = 0;
                return;
            }
        }
        equal_strides_ = true;
        common_stride_ = strides_[0];
    }

    bool equal_strides_{};
    size_type num_batch_entries_{};
    size_type common_stride_{};
    std::vector<size_type> strides_{};
};


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
    BatchLinOp* apply(const BatchLinOp* b, BatchLinOp* x)
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
    const BatchLinOp* apply(const BatchLinOp* b, BatchLinOp* x) const
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
    BatchLinOp* apply(const BatchLinOp* alpha, const BatchLinOp* b,
                      const BatchLinOp* beta, BatchLinOp* x)
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
    const BatchLinOp* apply(const BatchLinOp* alpha, const BatchLinOp* b,
                            const BatchLinOp* beta, BatchLinOp* x) const
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
    void set_size(const batch_dim<2>& size) { size_ = size; }

    /**
     * Returns the size of the batch operator.
     *
     * @return size of the batch operator
     */
    const batch_dim<2>& get_size() const noexcept { return size_; }

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
                        const dim<2>& size = dim<2>{})
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
                        const std::vector<dim<2>>& batch_sizes)
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
                        const batch_dim<2>& batch_sizes)
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
    virtual void apply_impl(const BatchLinOp* b, BatchLinOp* x) const = 0;

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
    virtual void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                            const BatchLinOp* beta, BatchLinOp* x) const = 0;

    /**
     * Throws a DimensionMismatch exception if the parameters to `apply` are of
     * the wrong size.
     *
     * @param b  batch vector(s) on which the operator is applied
     * @param x  output batch vector(s)
     */
    void validate_application_parameters(const BatchLinOp* b,
                                         const BatchLinOp* x) const
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
    void validate_application_parameters(const BatchLinOp* alpha,
                                         const BatchLinOp* b,
                                         const BatchLinOp* beta,
                                         const BatchLinOp* x) const
    {
        this->validate_application_parameters(b, x);
        GKO_ASSERT_BATCH_EQUAL_ROWS(
            alpha, batch_dim<2>(b->get_num_batch_entries(), dim<2>(1, 1)));
        GKO_ASSERT_BATCH_EQUAL_ROWS(
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

    const ConcreteBatchLinOp* apply(const BatchLinOp* b, BatchLinOp* x) const
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

    ConcreteBatchLinOp* apply(const BatchLinOp* b, BatchLinOp* x)
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

    const ConcreteBatchLinOp* apply(const BatchLinOp* alpha,
                                    const BatchLinOp* b, const BatchLinOp* beta,
                                    BatchLinOp* x) const
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

    ConcreteBatchLinOp* apply(const BatchLinOp* alpha, const BatchLinOp* b,
                              const BatchLinOp* beta, BatchLinOp* x)
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
        const std::vector<matrix_data<ValueType, IndexType>>& data) = 0;

    /**
     * Reads a matrix from a std::vector of matrix_assembly_data objects.
     *
     * @param data  the std::vector of matrix_assembly_data objects
     */
    void read(const std::vector<matrix_assembly_data<ValueType, IndexType>>&
                  assembly_data)
    {
        auto mat_data = std::vector<matrix_data<ValueType, IndexType>>(
            assembly_data.size());
        size_type ind = 0;
        for (const auto& i : assembly_data) {
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
        std::vector<matrix_data<ValueType, IndexType>>& data) const = 0;
};


/**
 * Adds the operation M <- a I + b M for batch matrix M,
 * batch identity operator I and batch scalars a and b.
 * M is the calling object.
 */
class BatchScaledIdentityAddable {
public:
    /**
     * Scales this and adds another scalar times the identity to it.
     *
     * @param a  Scalar to multiply the identity operator by before adding.
     * @param b  Scalar to multiply this before adding the scaled identity to
     *   it.
     */
    virtual void add_scaled_identity(const BatchLinOp* a, const BatchLinOp* b)
    {
        GKO_ASSERT_IS_BATCH_SCALAR(a);
        GKO_ASSERT_IS_BATCH_SCALAR(b);
        auto ae = make_temporary_clone(as<BatchLinOp>(this)->get_executor(), a);
        auto be = make_temporary_clone(as<BatchLinOp>(this)->get_executor(), b);
        add_scaled_identity_impl(ae.get(), be.get());
    }

private:
    virtual void add_scaled_identity_impl(const BatchLinOp* a,
                                          const BatchLinOp* b) = 0;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_LIN_OP_HPP_
