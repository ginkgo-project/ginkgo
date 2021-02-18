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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace matrix {


/**
 * BatchDense is a matrix format which explicitly stores all values of the
 * matrix.
 *
 * The values are stored in row-major format (values belonging to the same row
 * appear consecutive in the memory). Optionally, rows can be padded for better
 * memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 * @ingroup batch_dense
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class BatchDense : public EnableBatchLinOp<BatchDense<ValueType>>,
                   public EnableCreateMethod<BatchDense<ValueType>>,
                   public ConvertibleTo<BatchDense<next_precision<ValueType>>>,
                   public BatchReadableFromMatrixData<ValueType, int32>,
                   public BatchReadableFromMatrixData<ValueType, int64>,
                   public BatchWritableToMatrixData<ValueType, int32>,
                   public BatchWritableToMatrixData<ValueType, int64>,
                   public BatchTransposable {
    friend class EnableCreateMethod<BatchDense>;
    friend class EnablePolymorphicObject<BatchDense, BatchLinOp>;
    friend class BatchDense<to_complex<ValueType>>;

public:
    using EnableBatchLinOp<BatchDense>::convert_to;
    using EnableBatchLinOp<BatchDense>::move_to;
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int64;
    using transposed_type = BatchDense<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchDense>;
    using complex_type = to_complex<BatchDense>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a BatchDense matrix with the configuration of another BatchDense
     * matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchDense> create_with_config_of(
        const BatchDense *other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    friend class BatchDense<next_precision<ValueType>>;

    void convert_to(
        BatchDense<next_precision<ValueType>> *result) const override;

    void move_to(BatchDense<next_precision<ValueType>> *result) override;

    void read(std::vector<const mat_data> &data) override;

    void read(std::vector<const mat_data32> &data) override;

    void write(std::vector<mat_data> &data) const override;

    void write(std::vector<mat_data32> &data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc get_values()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * Returns the stride of the matrix.
     *
     * @return the stride of the matrix.
     */
    const std::vector<size_type> &get_strides() const noexcept
    {
        return strides_;
    }

    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Returns a single element of the matrix.
     *
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type &at(size_type batch, size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    /**
     * @copydoc BatchDense::at(size_type, size_type)
     */
    value_type at(size_type batch, size_type row, size_type col) const noexcept
    {
        return values_.get_const_data()[linearize_index(row, col)];
    }

    /**
     * Scales the matrix with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 BatchDense matrix, the entire matrix is
     * scaled by alpha. If it is a BatchDense row vector of values, then i-th
     * column of the matrix is scaled with the i-th element of alpha (the number
     * of columns of alpha has to match the number of columns of the matrix).
     */
    void scale(const BatchLinOp *alpha)
    {
        auto exec = this->get_executor();
        this->scale_impl(make_temporary_clone(exec, alpha).get());
    }

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 BatchDense matrix, the entire matrix is
     * scaled by alpha. If it is a BatchDense row vector of values, then i-th
     * column of the matrix is scaled with the i-th element of alpha (the number
     * of columns of alpha has to match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    void add_scaled(const BatchLinOp *alpha, const BatchLinOp *b)
    {
        auto exec = this->get_executor();
        this->add_scaled_impl(make_temporary_clone(exec, alpha).get(),
                              make_temporary_clone(exec, b).get());
    }

    /**
     * Computes the column-wise dot product of this matrix and `b`. The
     * conjugate of this is taken.
     *
     * @param b  a BatchDense matrix of same dimension as this
     * @param result  a BatchDense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_dot(const BatchLinOp *b, BatchLinOp *result) const
    {
        auto exec = this->get_executor();
        this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                               make_temporary_clone(exec, result).get());
    }

    /**
     * Computes the Euclidian (L^2) norm of this matrix.
     *
     * @param result  a BatchDense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(BatchLinOp *result) const
    {
        auto exec = this->get_executor();
        this->compute_norm2_impl(make_temporary_clone(exec, result).get());
    }

private:
    const size_type compute_batch_mem(const std::vector<dim<2>> sizes,
                                      const std::vector<size_type> strides)
    {
        GKO_ASSERT(sizes.size() == strides.size());
        size_type mem_req = 0;
        for (auto i = 0; i < sizes.size(); ++i) {
            mem_req += (sizes[i])[0] * strides[i];
        }
        return mem_req;
    }

    const std::vector<size_type> extract_nth_dim(
        const int dim, const std::vector<gko::dim<2>> sizes)
    {
        // auto ndim_vec = std::vector<size_type>{};
        auto ndim_vec = std::vector<size_type>(sizes.size(), 0);
        for (auto i = 0; i < sizes.size(); ++i) {
            ndim_vec[i] = (sizes[i])[dim];
        }
        return ndim_vec;
    }

    const std::vector<size_type> compute_num_elems_per_batch(
        const std::vector<gko::dim<2>> sizes,
        const std::vector<size_type> strides)
    {
        // auto num_elems = std::vector<size_type>{};
        auto num_elems = std::vector<size_type>(sizes.size(), 0);
        for (auto i = 0; i < sizes.size(); ++i) {
            num_elems[i] = (sizes[i])[0] * strides[i];
        }
        return num_elems;
    }

protected:
    /**
     * Creates an uninitialized BatchDense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    BatchDense(std::shared_ptr<const Executor> exec,
               const std::vector<dim<2>> sizes = std::vector<dim<2>>{})
        : BatchDense(std::move(exec), sizes, extract_nth_dim(1, sizes))
    {}

    /**
     * Creates an uninitialized BatchDense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param sizes  sizes of the batch matrices in a std::vector
     * @param strides  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    BatchDense(std::shared_ptr<const Executor> exec,
               const std::vector<dim<2>> sizes,
               const std::vector<size_type> strides)
        : EnableBatchLinOp<BatchDense>(exec, sizes),
          values_(exec, compute_batch_mem(sizes, strides)),
          strides_(strides)
    {
        num_elems_per_batch_ = compute_num_elems_per_batch(sizes, strides);
    }

    /**
     * Creates a BatchDense matrix from an already allocated (and initialized)
     * array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param sizes  sizes of the batch matrices in a std::vector
     * @param values  array of matrix values
     * @param strides  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    BatchDense(std::shared_ptr<const Executor> exec,
               const std::vector<dim<2>> sizes, ValuesArray &&values,
               const std::vector<size_type> strides)
        : EnableBatchLinOp<BatchDense>(exec, sizes),
          values_{exec, std::forward<ValuesArray>(values)},
          strides_{strides},
          num_elems_per_batch_(compute_num_elems_per_batch(sizes, strides))
    {
        GKO_ENSURE_IN_BOUNDS(
            std::accumulate(num_elems_per_batch_.begin(),
                            num_elems_per_batch_.begin() + this->num_batches_,
                            0) -
                1,
            values_.get_num_elems());
    }

    /**
     * Creates a BatchDense matrix with the same configuration as the callers
     * matrix.
     *
     * @returns a BatchDense matrix with the same configuration as the caller.
     */
    virtual std::unique_ptr<BatchDense> create_with_same_config() const
    {
        return BatchDense::create(this->get_executor(), this->get_sizes(),
                                  this->get_strides());
    }

    /**
     * @copydoc scale(const LinOp *)
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of scale(const LinOp *alpha).
     */
    virtual void scale_impl(const BatchLinOp *alpha);

    /**
     * @copydoc add_scaled(const LinOp *, const LinOp *)
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of add_scale(const LinOp *alpha, const LinOp *b).
     */
    virtual void add_scaled_impl(const BatchLinOp *alpha, const BatchLinOp *b);

    /**
     * @copydoc compute_dot(const LinOp *, LinOp *) const
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of compute_dot(const LinOp *b, LinOp *result).
     */
    virtual void compute_dot_impl(const BatchLinOp *b,
                                  BatchLinOp *result) const;

    /**
     * @copydoc compute_norm2(LinOp *) const
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of compute_norm2(LinOp *result).
     */
    virtual void compute_norm2_impl(BatchLinOp *result) const;

    void apply_impl(const BatchLinOp *b, BatchLinOp *x) const override;

    void apply_impl(const BatchLinOp *alpha, const BatchLinOp *b,
                    const BatchLinOp *beta, BatchLinOp *x) const override;

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row * strides_[0] + col;
    }

private:
    Array<value_type> values_;
    std::vector<size_type> strides_;
    std::vector<size_type> num_elems_per_batch_;
};


}  // namespace matrix


/**
 * Creates and initializes a column-vector.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row stride for the temporary Dense matrix
 * @param vals  values used to initialize the vector
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
    size_type stride, std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    auto tmp =
        batch_dense::create(exec->get_master(), dim<2>{num_rows, 1}, stride);
    size_type idx = 0;
    for (const auto &elem : vals) {
        tmp->at(idx) = elem;
        ++idx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}

/**
 * Creates and initializes a column-vector.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the matrix to the requested type. The stride of
 * the intermediate Dense matrix is set to 1.
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
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    return batch_initialize<Matrix>(1, vals, std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a matrix.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row stride for the temporary Dense matrix
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    size_type stride,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    size_type num_cols = num_rows > 0 ? begin(vals)->size() : 1;
    auto tmp = batch_dense::create(exec->get_master(),
                                   dim<2>{num_rows, num_cols}, stride);
    size_type ridx = 0;
    for (const auto &row : vals) {
        size_type cidx = 0;
        for (const auto &elem : row) {
            tmp->at(ridx, cidx) = elem;
            ++cidx;
        }
        ++ridx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}


/**
 * Creates and initializes a matrix.
 *
 * This function first creates a temporary Dense matrix, fills it with passed in
 * values, and then converts the matrix to the requested type. The stride of
 * the intermediate Dense matrix is set to the number of columns of the
 * initializer list.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    return batch_initialize<Matrix>(vals.size() > 0 ? begin(vals)->size() : 0,
                                    vals, std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
