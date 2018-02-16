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
#include "core/base/convertible.hpp"
#include "core/base/executor.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"
#include "core/base/types.hpp"


#include <initializer_list>


namespace gko {
namespace matrix {


template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class Ell;

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
class Dense : public BasicLinOp<Dense<ValueType>>,
              public ConvertibleTo<Csr<ValueType, int32>>,
              public ConvertibleTo<Csr<ValueType, int64>>,
              public ConvertibleTo<Ell<ValueType, int32>>,
              public ConvertibleTo<Ell<ValueType, int64>>,
              public ReadableFromMtx,
              public Transposable {
    friend class BasicLinOp<Dense>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Ell<ValueType, int32>;
    friend class Ell<ValueType, int64>;

public:
    using BasicLinOp<Dense>::create;
    using BasicLinOp<Dense>::convert_to;
    using BasicLinOp<Dense>::move_to;

    using value_type = ValueType;

    /**
     * Creates a Dense matrix with the configuration of another Dense matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<Dense> create_with_config_of(const Dense *other)
    {
        return create(other->get_executor(), other->get_num_rows(),
                      other->get_num_cols(), other->get_padding());
    }

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    void convert_to(Csr<ValueType, int32> *result) const override;

    void move_to(Csr<ValueType, int32> *result) override;

    void convert_to(Csr<ValueType, int64> *result) const override;

    void move_to(Csr<ValueType, int64> *result) override;

    void convert_to(Ell<ValueType, int32> *result) const override;

    void move_to(Ell<ValueType, int32> *result) override;

    void convert_to(Ell<ValueType, int64> *result) const override;

    void move_to(Ell<ValueType, int64> *result) override;

    void convert_to(Ell<ValueType, int32> *result, const size_type padding) const;

    void move_to(Ell<ValueType, int32> *result, const size_type padding);

    void convert_to(Ell<ValueType, int64> *result, const size_type padding) const;

    void move_to(Ell<ValueType, int64> *result, const size_type padding);

    void read_from_mtx(const std::string &filename) override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return  the pointer to the array of values
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
     * Returns the padding of the matrix.
     */
    size_type get_padding() const noexcept { return padding_; }

    /**
     * Returns a single element of the matrix.
     *
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
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
     *             (ignoring the padding)
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
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
     * @param b  a Dense matrix of same dimensions as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    virtual void compute_dot(const LinOp *b, LinOp *result) const;

protected:
    /**
     * Creates an empty Dense matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Dense(std::shared_ptr<const Executor> exec)
        : BasicLinOp<Dense>(exec, 0, 0, 0), values_(exec)
    {}

    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows  number of rows
     * @param num_cols  number of columns
     * @param padding  padding of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    Dense(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols, size_type padding)
        : BasicLinOp<Dense>(exec, num_rows, num_cols, num_rows * padding),
          values_(exec, num_rows * padding),
          padding_(padding)
    {}

    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows  number of rows
     * @param num_cols  number of columns
     */
    Dense(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols)
        : Dense(std::move(exec), num_rows, num_cols, num_cols)
    {}

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row * padding_ + col;
    }

    size_type linearize_index(size_type idx) const noexcept
    {
        return linearize_index(idx / this->get_num_cols(),
                               idx % this->get_num_cols());
    }

private:
    Array<value_type> values_;
    size_type padding_{};
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
 * @param padding  row padding for the temporary Dense matrix
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type padding, std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    int num_rows = vals.size();
    auto tmp = dense::create(exec->get_master(), num_rows, 1, padding);
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
 * values, and then converts the matrix to the requested type. The padding of
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
 * @param padding  row padding for the temporary Dense matrix
 * @param vals  values used to initialize the matrix
 * @param exec  Executor associated to the matrix
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type padding,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs &&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    int num_rows = vals.size();
    int num_cols = num_rows > 0 ? begin(vals)->size() : 1;
    auto tmp = dense::create(exec->get_master(), num_rows, num_cols, padding);
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
 * values, and then converts the matrix to the requested type. The padding of
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
