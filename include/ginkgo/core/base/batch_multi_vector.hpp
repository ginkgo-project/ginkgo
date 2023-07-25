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

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/batch_lin_op_helpers.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {

/**
 * BatchMultiVector is a batch matrix format which explicitly stores all values
 * of the vector in each of the batches.
 *
 * The values in each of the batches are stored in row-major format (values
 * belonging to the same row appear consecutive in the memory).
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @ingroup batch_multi_vector
 */
template <typename ValueType = default_precision>
class BatchMultiVector
    : public EnablePolymorphicObject<BatchMultiVector<ValueType>>,
      public EnablePolymorphicAssignment<BatchMultiVector<ValueType>>,
      public EnableCreateMethod<BatchMultiVector<ValueType>>,
      public ConvertibleTo<BatchMultiVector<next_precision<ValueType>>>,
      public BatchReadableFromMatrixData<ValueType, int32>,
      public BatchReadableFromMatrixData<ValueType, int64>,
      public BatchWritableToMatrixData<ValueType, int32>,
      public BatchWritableToMatrixData<ValueType, int64> {
    friend class EnableCreateMethod<BatchMultiVector>;
    friend class EnablePolymorphicObject<BatchMultiVector>;
    friend class BatchMultiVector<to_complex<ValueType>>;

public:
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;
    using EnablePolymorphicAssignment<BatchMultiVector>::convert_to;
    using EnablePolymorphicAssignment<BatchMultiVector>::move_to;
    using ConvertibleTo<
        BatchMultiVector<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<BatchMultiVector<next_precision<ValueType>>>::move_to;

    using value_type = ValueType;
    using index_type = int32;
    using unbatch_type = matrix::Dense<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchMultiVector<ValueType>>;
    using complex_type = to_complex<BatchMultiVector<ValueType>>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a BatchMultiVector matrix with the configuration of another
     * BatchMultiVector matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchMultiVector> create_with_config_of(
        ptr_param<const BatchMultiVector> other);

    friend class BatchMultiVector<next_precision<ValueType>>;

    void convert_to(
        BatchMultiVector<next_precision<ValueType>>* result) const override;

    void move_to(BatchMultiVector<next_precision<ValueType>>* result) override;

    void read(const std::vector<mat_data>& data) override;

    void read(const std::vector<mat_data32>& data) override;

    void write(std::vector<mat_data>& data) const override;

    void write(std::vector<mat_data32>& data) const override;

    /**
     * Unbatches the batched dense and creates a std::vector of Dense matrices
     *
     * @return  a std::vector containing the Dense matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const;

    /**
     * Returns the batch size.
     *
     * @return the batch size
     */
    batch_dim<2> get_size() const { return batch_size_; }

    /**
     * Returns the number of batch entries.
     *
     * @return the number of batch entries
     */
    size_type get_num_batch_entries() const
    {
        return batch_size_.get_num_batch_entries();
    }

    /**
     * Returns the common size of the batch entries.
     *
     * @return the common size stored
     */
    dim<2> get_common_size() const { return batch_size_.get_common_size(); }

    /**
     * Returns a pointer to the array of values of the vector.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * Returns a pointer to the array of values of the vector.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_data() +
               this->get_size().get_cumulative_offset(batch);
    }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * @copydoc get_values(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_const_data() +
               this->get_size().get_cumulative_offset(batch);
    }

    /**
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batches.
     *
     * @return the number of elements explicitly stored in the vector,
     *         cumulative across all the batches
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Returns a single element for a particular batch.
     *
     * @param batch  the batch index to be queried
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& at(size_type batch, size_type row, size_type col)
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_data()[linearize_index(batch, row, col)];
    }

    /**
     * @copydoc BatchMultiVector::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch, size_type row, size_type col) const
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_const_data()[linearize_index(batch, row, col)];
    }

    /**
     * Returns a single element for a particular batch entry.
     *
     * Useful for iterating across all elements of the vector.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param batch  the batch index to be queried
     * @param idx  a linear index of the requested element
     *             (ignoring the stride)
     *
     * @note  the method has to be called on the same Executor the vector is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    ValueType& at(size_type batch, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch, idx)];
    }

    /**
     * @copydoc BatchMultiVector::at(size_type, size_type, size_type)
     */
    ValueType at(size_type batch, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch, idx)];
    }

    /**
     * Scales the vector with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 BatchMultiVector matrix, the entire matrix
     * (all batches) is scaled by alpha. If it is a BatchMultiVector row vector
     * of values, then i-th column of the vector is scaled with the i-th element
     * of alpha (the number of columns of alpha has to match the number of
     * columns of the matrix).
     */
    void scale(ptr_param<const BatchMultiVector<ValueType>> alpha);

    /**
     * Adds `b` scaled by `alpha` to the vector (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 BatchMultiVector matrix, the entire matrix
     * is scaled by alpha. If it is a BatchMultiVector row vector of values,
     * then i-th column of the vector is scaled with the i-th element of alpha
     * (the number of columns of alpha has to match the number of columns of the
     * vector).
     * @param b  a matrix of the same dimension as this
     */
    void add_scaled(ptr_param<const BatchMultiVector<ValueType>> alpha,
                    ptr_param<const BatchMultiVector<ValueType>> b);

    /**
     * Computes the column-wise dot product of each matrix in this batch and its
     * corresponding entry in `b`.
     *
     * @param b  a BatchMultiVector matrix of same dimension as this
     * @param result  a BatchMultiVector row vector, used to store the dot
     * product (the number of column in the vector must match the number of
     * columns of this)
     */
    void compute_dot(ptr_param<const BatchMultiVector<ValueType>> b,
                     ptr_param<BatchMultiVector<ValueType>> result) const;

    /**
     * Computes the column-wise conjugate dot product of each matrix in this
     * batch and its corresponding entry in `b`. If the vector has complex
     * value_type, then the conjugate of this is taken.
     *
     * @param b  a BatchMultiVector matrix of same dimension as this
     * @param result  a BatchMultiVector row vector, used to store the dot
     * product (the number of column in the vector must match the number of
     * columns of this)
     */
    void compute_conj_dot(ptr_param<const BatchMultiVector<ValueType>> b,
                          ptr_param<BatchMultiVector<ValueType>> result) const;

    /**
     * Computes the Euclidean (L^2) norm of each matrix in this batch.
     *
     * @param result  a BatchMultiVector row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(
        ptr_param<BatchMultiVector<remove_complex<ValueType>>> result) const;

    /**
     * Creates a constant (immutable) batch dense matrix from a constant array.
     *
     * @param exec  the executor to create the vector on
     * @param size  the dimensions of the vector
     * @param values  the value array of the vector
     * @param stride  the row-stride of the vector
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the vector) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const BatchMultiVector<ValueType>> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& values);

    /**
     * Fills the input BatchMultiVector with a given value
     *
     * @param value  the value to be filled
     */
    void fill(ValueType value);

private:
    inline batch_dim<2> compute_batch_size(
        const std::vector<matrix::Dense<ValueType>*>& matrices)
    {
        auto common_size = matrices[0]->get_size();
        for (int i = 1; i < matrices.size(); ++i) {
            GKO_ASSERT_EQUAL_DIMENSIONS(common_size, matrices[i]->get_size());
        }
        return batch_dim<2>{matrices.size(), common_size};
    }

    inline size_type compute_num_elems(const batch_dim<2>& size)
    {
        return size.get_cumulative_offset(size.get_num_batch_entries());
    }


protected:
    /**
     * Sets the size of the BatchMultiVector.
     *
     * @param value  the new size of the operator
     */
    void set_size(const batch_dim<2>& value) noexcept;

    /**
     * Creates an uninitialized BatchMultiVector matrix of the specified size.
     *
     * @param exec  Executor associated to the vector
     * @param size  size of the vector
     */
    BatchMultiVector(std::shared_ptr<const Executor> exec,
                     const batch_dim<2>& size = batch_dim<2>{})
        : EnablePolymorphicObject<BatchMultiVector<ValueType>>(exec),
          batch_size_(size),
          values_(exec, compute_num_elems(size))
    {}

    /**
     * Creates a BatchMultiVector matrix from an already allocated (and
     * initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the vector
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param values  array of matrix values
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the vector.
     */
    template <typename ValuesArray>
    BatchMultiVector(std::shared_ptr<const Executor> exec,
                     const batch_dim<2>& size, ValuesArray&& values)
        : EnablePolymorphicObject<BatchMultiVector<ValueType>>(exec),
          batch_size_(size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        // Ensure that the values array has the correct size
        auto num_elems = compute_num_elems(size);
        GKO_ENSURE_IN_BOUNDS(num_elems, values_.get_num_elems() + 1);
    }

    /**
     * Creates a BatchMultiVector matrix from a vector of matrices
     *
     * @param exec  Executor associated to the vector
     * @param matrices  The matrices that need to be batched.
     */
    BatchMultiVector(std::shared_ptr<const Executor> exec,
                     const std::vector<matrix::Dense<ValueType>*>& matrices)
        : EnablePolymorphicObject<BatchMultiVector<ValueType>>(exec),
          batch_size_{compute_batch_size(matrices)},
          values_(exec, compute_num_elems(batch_size_))
    {
        for (size_type i = 0; i < this->get_num_batch_entries(); ++i) {
            auto local_exec = matrices[i]->get_executor();
            exec->copy_from(
                local_exec.get(), matrices[i]->get_num_stored_elements(),
                matrices[i]->get_const_values(),
                this->get_values() + this->get_size().get_cumulative_offset(i));
        }
    }

    /**
     * Creates a BatchMultiVector matrix by duplicating BatchMultiVector matrix
     *
     * @param exec  Executor associated to the vector
     * @param num_duplications  The number of times to duplicate
     * @param input  the vector to be duplicated.
     */
    BatchMultiVector(std::shared_ptr<const Executor> exec,
                     size_type num_duplications,
                     const BatchMultiVector<value_type>* input)
        : BatchMultiVector<ValueType>(
              exec, gko::batch_dim<2>(
                        input->get_num_batch_entries() * num_duplications,
                        input->get_common_size()))
    {
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_values(), this->get_values() + offset);
            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchMultiVector matrix by duplicating Dense matrix
     *
     * @param exec  Executor associated to the vector
     * @param num_duplications  The number of times to duplicate
     * @param input  the vector to be duplicated.
     */
    BatchMultiVector(std::shared_ptr<const Executor> exec,
                     size_type num_duplications,
                     const matrix::Dense<value_type>* input)
        : BatchMultiVector<ValueType>(
              exec, gko::batch_dim<2>(num_duplications, input->get_size()))
    {
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_values(), this->get_values() + offset);
            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchMultiVector matrix with the same configuration as the
     * callers matrix.
     *
     * @returns a BatchMultiVector matrix with the same configuration as the
     * caller.
     */
    std::unique_ptr<BatchMultiVector> create_with_same_config() const;

    size_type linearize_index(size_type batch, size_type row,
                              size_type col) const noexcept
    {
        return batch_size_.get_cumulative_offset(batch) +
               row * batch_size_.get_common_size()[1] + col;
    }

    size_type linearize_index(size_type batch, size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_common_size()[1],
                               idx % this->get_common_size()[1]);
    }

private:
    batch_dim<2> batch_size_;
    array<value_type> values_;
};


/**
 * Creates and initializes a batch of column-vectors.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the vector to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the batch vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchMultiVector
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_multi_vector = BatchMultiVector<typename Matrix::value_type>;
    size_type num_batch_entries = vals.size();
    GKO_ASSERT(num_batch_entries > 0);
    auto vals_begin = begin(vals);
    size_type common_num_rows = vals_begin->size();
    auto common_size = dim<2>(common_num_rows, 1);
    for (auto& val : vals) {
        GKO_ASSERT_EQ(common_num_rows, val.size());
    }
    auto b_size = batch_dim<2>(num_batch_entries, common_size);
    auto tmp = batch_multi_vector::create(exec->get_master(), b_size);
    size_type batch = 0;
    for (const auto& b : vals) {
        size_type idx = 0;
        for (const auto& elem : b) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
        ++batch;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


/**
 * Creates and initializes a batch of matrices.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the vector to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchMultiVector
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<std::initializer_list<
        std::initializer_list<typename Matrix::value_type>>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_multi_vector = BatchMultiVector<typename Matrix::value_type>;
    size_type num_batch_entries = vals.size();
    GKO_ASSERT(num_batch_entries > 0);
    auto vals_begin = begin(vals);
    size_type common_num_rows = vals_begin->size();
    size_type common_num_cols = vals_begin->begin()->size();
    auto common_size = dim<2>(common_num_rows, common_num_cols);
    for (const auto& b : vals) {
        auto num_rows = b.size();
        auto num_cols = begin(b)->size();
        auto b_size = dim<2>(num_rows, num_cols);
        GKO_ASSERT_EQUAL_DIMENSIONS(b_size, common_size);
    }

    auto b_size = batch_dim<2>(num_batch_entries, common_size);
    auto tmp = batch_multi_vector::create(exec->get_master(), b_size);
    size_type batch = 0;
    for (const auto& b : vals) {
        size_type ridx = 0;
        for (const auto& row : b) {
            size_type cidx = 0;
            for (const auto& elem : row) {
                tmp->at(batch, ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
        ++batch;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


/**
 * Creates and initializes a batch column-vector by making copies of the single
 * input column vector.
 *
 * This function first creates a temporary batch dense matrix, fills it with
 * passed in values, and then converts the vector to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                  interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param num_vectors  The number of times the input vector is copied into
 *                     the final output
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchMultiVector
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    const size_type num_vectors,
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_multi_vector = BatchMultiVector<typename Matrix::value_type>;
    size_type num_batch_entries = num_vectors;
    GKO_ASSERT(num_batch_entries > 0);
    auto b_size = batch_dim<2>(num_batch_entries, dim<2>(vals.size(), 1));
    auto tmp = batch_multi_vector::create(exec->get_master(), b_size);
    for (size_type batch = 0; batch < num_vectors; batch++) {
        size_type idx = 0;
        for (const auto& elem : vals) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


/**
 * Creates and initializes a matrix from copies of a given matrix.
 *
 * This function first creates a temporary batch dense matrix, fills it with
 * passed in values, and then converts the vector to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row strides for the temporary batch dense matrix
 * @param num_matrices  The number of times the input matrix is copied into
 *                     the final output
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    const size_type num_batch_entries,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_multi_vector = BatchMultiVector<typename Matrix::value_type>;
    GKO_ASSERT(num_batch_entries > 0);
    auto common_size = dim<2>(vals.size(), begin(vals)->size());
    batch_dim<2> b_size(num_batch_entries, common_size);
    auto tmp = batch_multi_vector::create(exec->get_master(), b_size);
    for (size_type batch = 0; batch < num_batch_entries; batch++) {
        size_type ridx = 0;
        for (const auto& row : vals) {
            size_type cidx = 0;
            for (const auto& elem : row) {
                tmp->at(batch, ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_BATCH_MULTI_VECTOR_HPP_
