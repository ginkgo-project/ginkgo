/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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

#ifndef GKO_CORE_MATRIX_DENSE_HPP_
#define GKO_CORE_MATRIX_DENSE_HPP_


#include "core/base/array.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_io.hpp"
#include "core/base/range_accessors.hpp"
#include "core/base/types.hpp"


#include <initializer_list>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;


/**
 * Dense is a matrix format which explicitly stores all values of the matrix.
 *
 * The values are stored in row-major format (values belonging to the same row
 * appear consecutive in the memory). Optionally, rows can be padded for better
 * memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 */
template <typename ValueType = default_precision>
class Dense : public EnableLinOp<Dense<ValueType>>,
              public EnableCreateMethod<Dense<ValueType>>,
              public ConvertibleTo<Coo<ValueType, int32>>,
              public ConvertibleTo<Coo<ValueType, int64>>,
              public ConvertibleTo<Csr<ValueType, int32>>,
              public ConvertibleTo<Csr<ValueType, int64>>,
              public ConvertibleTo<Ell<ValueType, int32>>,
              public ConvertibleTo<Ell<ValueType, int64>>,
              public ConvertibleTo<Hybrid<ValueType, int32>>,
              public ConvertibleTo<Hybrid<ValueType, int64>>,
              public ConvertibleTo<Sellp<ValueType, int32>>,
              public ConvertibleTo<Sellp<ValueType, int64>>,
              public ReadableFromMatrixData<ValueType, int32>,
              public ReadableFromMatrixData<ValueType, int64>,
              public WritableToMatrixData<ValueType, int32>,
              public WritableToMatrixData<ValueType, int64>,
              public Transposable {
    friend class EnableCreateMethod<Dense>;
    friend class EnablePolymorphicObject<Dense, LinOp>;
    friend class Coo<ValueType, int32>;
    friend class Coo<ValueType, int64>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Ell<ValueType, int32>;
    friend class Ell<ValueType, int64>;
    friend class Hybrid<ValueType, int32>;
    friend class Hybrid<ValueType, int64>;
    friend class Sellp<ValueType, int32>;
    friend class Sellp<ValueType, int64>;

public:
    using EnableLinOp<Dense>::convert_to;
    using EnableLinOp<Dense>::move_to;

    using value_type = ValueType;
    using index_type = int64;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a Dense matrix with the configuration of another Dense matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<Dense> create_with_config_of(const Dense *other)
    {
        return Dense::create(other->get_executor(), other->get_size(),
                             other->get_stride());
    }

    void convert_to(Coo<ValueType, int32> *result) const override;

    void move_to(Coo<ValueType, int32> *result) override;

    void convert_to(Coo<ValueType, int64> *result) const override;

    void move_to(Coo<ValueType, int64> *result) override;

    void convert_to(Csr<ValueType, int32> *result) const override;

    void move_to(Csr<ValueType, int32> *result) override;

    void convert_to(Csr<ValueType, int64> *result) const override;

    void move_to(Csr<ValueType, int64> *result) override;

    void convert_to(Ell<ValueType, int32> *result) const override;

    void move_to(Ell<ValueType, int32> *result) override;

    void convert_to(Ell<ValueType, int64> *result) const override;

    void move_to(Ell<ValueType, int64> *result) override;

    void convert_to(Hybrid<ValueType, int32> *result) const override;

    void move_to(Hybrid<ValueType, int32> *result) override;

    void convert_to(Hybrid<ValueType, int64> *result) const override;

    void move_to(Hybrid<ValueType, int64> *result) override;

    void convert_to(Sellp<ValueType, int32> *result) const override;

    void move_to(Sellp<ValueType, int32> *result) override;

    void convert_to(Sellp<ValueType, int64> *result) const override;

    void move_to(Sellp<ValueType, int64> *result) override;

    void read(const mat_data &data) override;

    void read(const mat_data32 &data) override;

    void write(mat_data &data) const override;

    void write(mat_data32 &data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

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
    size_type get_stride() const noexcept { return stride_; }

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
    value_type &at(size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    /**
     * @copydoc Dense::at(size_type, size_type)
     */
    value_type at(size_type row, size_type col) const noexcept
    {
        return values_.get_const_data()[linearize_index(row, col)];
    }

    /**
     * Returns a single element of the matrix.
     *
     * Useful for iterating across all elements of the matrix.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param idx  a linear index of the requested element
     *             (ignoring the stride)
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    ValueType &at(size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(idx)];
    }

    /**
     * @copydoc Dense::at(size_type)
     */
    ValueType at(size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(idx)];
    }

    /**
     * Scales the matrix with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     */
    virtual void scale(const LinOp *alpha);

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    virtual void add_scaled(const LinOp *alpha, const LinOp *b);

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    virtual void compute_dot(const LinOp *b, LinOp *result) const;

    /**
     * Create a submatrix from the original matrix.
     * Warning: defining stride for this create_submatrix method might cause
     * wrong memory access. Better use the create_submatrix(rows, columns)
     * method instead.
     *
     * @param rows     row span
     * @param columns  column span
     * @param stride   stride of the new submatrix.
     */
    std::unique_ptr<Dense> create_submatrix(const span &rows,
                                            const span &columns,
                                            const size_type stride)
    {
        row_major_range range_this{this->get_values(), this->get_size()[0],
                                   this->get_size()[1], this->get_stride()};
        auto range_result = range_this(rows, columns);
        return Dense::create(
            this->get_executor(),
            dim<2>{range_result.length(0), range_result.length(1)},
            Array<ValueType>::view(
                this->get_executor(),
                range_result.length(0) * range_this.length(1),
                range_result->data),
            stride);
    }

    /*
     * Create a submatrix from the original matrix.
     *
     * @param rows     row span
     * @param columns  column span
     */
    std::unique_ptr<Dense> create_submatrix(const span &rows,
                                            const span &columns)
    {
        return create_submatrix(rows, columns, this->get_size()[1]);
    }

protected:
    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Dense(std::shared_ptr<const Executor> exec, const dim<2> &size = dim<2>{})
        : Dense(std::move(exec), size, size[1])
    {}

    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    Dense(std::shared_ptr<const Executor> exec, const dim<2> &size,
          size_type stride)
        : EnableLinOp<Dense>(exec, size),
          values_(exec, size[0] * stride),
          stride_(stride)
    {}

    /**
     * Creates a Dense matrix from an already allocated (and initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename ValuesArray>
    Dense(std::shared_ptr<const Executor> exec, const dim<2> &size,
          ValuesArray &&values, size_type stride)
        : EnableLinOp<Dense>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          stride_{stride}
    {
        ENSURE_IN_BOUNDS((size[0] - 1) * stride + size[1] - 1,
                         values_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row * stride_ + col;
    }

    size_type linearize_index(size_type idx) const noexcept
    {
        return linearize_index(idx / this->get_size()[1],
                               idx % this->get_size()[1]);
    }

private:
    Array<value_type> values_;
    size_type stride_;
};  // namespace matrix


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
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride, std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    auto tmp = dense::create(exec->get_master(), dim<2>{num_rows, 1}, stride);
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
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    return initialize<Matrix>(1, vals, std::move(exec),
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
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    size_type num_cols = num_rows > 0 ? begin(vals)->size() : 1;
    auto tmp =
        dense::create(exec->get_master(), dim<2>{num_rows, num_cols}, stride);
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
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    return initialize<Matrix>(vals.size() > 0 ? begin(vals)->size() : 0, vals,
                              std::move(exec),
                              std::forward<TArgs>(create_args)...);
}


}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_HPP_
