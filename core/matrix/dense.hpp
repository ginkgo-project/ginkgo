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


#include <initializer_list>


namespace gko {
namespace matrix {


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
class Dense : public LinOp, public ConvertibleTo<Dense<ValueType>> {
public:
    using value_type = ValueType;

    /**
     * Creates an empty Dense matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec)
    {
        return create(exec, 0, 0, 0);
    }

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
    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         size_type num_rows, size_type num_cols,
                                         size_type padding)
    {
        return std::unique_ptr<Dense>(
            new Dense(std::move(exec), num_rows, num_cols, padding));
    }

    /**
     * Creates and initializes a Dense column-vector.
     *
     * @param exec  Executor associated to the vector
     * @param padding  padding of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         size_type padding,
                                         std::initializer_list<ValueType> vals)
    {
        int num_rows = vals.size();
        std::unique_ptr<Dense> tmp(
            new Dense(exec->get_master(), num_rows, 1, padding));
        size_type idx = 0;
        for (const auto &elem : vals) {
            tmp->at(idx) = elem;
            ++idx;
        }
        auto result = create(std::move(exec));
        result->copy_from(std::move(tmp));
        return result;
    }

    /**
     * Creates and initializes a Dense column-vector.
     *
     * The padding of the vector is set to 1.
     *
     * @param exec  Executor associated to the vector
     * @param vals  values used to initialize the vector
     */
    static std::unique_ptr<Dense> create(std::shared_ptr<const Executor> exec,
                                         std::initializer_list<ValueType> vals)
    {
        return create(std::move(exec), 1, vals);
    }

    /**
     * Creates and initializes a Dense matrix.
     *
     * @param exec  Executor associated to the matrix
     * @param padding  padding of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     * @param vals  values used to initialize the matrix
     */
    static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec, size_type padding,
        std::initializer_list<std::initializer_list<ValueType>> vals)
    {
        int num_rows = vals.size();
        int num_cols = num_rows > 0 ? begin(vals)->size() : 1;
        std::unique_ptr<Dense> tmp(
            new Dense(exec->get_master(), num_rows, num_cols, padding));
        size_type ridx = 0;
        for (const auto &row : vals) {
            size_type cidx = 0;
            for (const auto &elem : row) {
                tmp->at(ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
        auto result = create(std::move(exec));
        result->copy_from(std::move(tmp));
        return result;
    }

    /**
     * Creates and initializes a Dense matrix.
     *
     * Padding is set to the number of columns of the matrix.
     *
     * @param exec  Executor associated to the matrix
     * @param vals  values used to initialize the matrix
     */
    static std::unique_ptr<Dense> create(
        std::shared_ptr<const Executor> exec,
        std::initializer_list<std::initializer_list<ValueType>> vals)
    {
        using std::max;
        return create(
            std::move(exec),
            vals.size() > 0 ? max<size_type>(begin(vals)->size(), 1) : 1, vals);
    }

    /**
     * Gets the array containing the values of the matrix.
     */
    Array<value_type> &get_values() noexcept { return values_; }

    /**
     * Gets the array containing the values of the matrix.
     */
    const Array<value_type> &get_values() const noexcept { return values_; }

    /**
     * Returns the padding of the matrix.
     */
    size_type get_padding() const { return padding_; }

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
    ValueType &at(size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    /**
     * @copydoc Dense::at(size_type, size_type)
     */
    ValueType at(size_type row, size_type col) const noexcept
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

    void copy_from(const LinOp *other) override;

    void copy_from(std::unique_ptr<LinOp> other) override;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    std::unique_ptr<LinOp> clone_type() const override;

    void clear() override;

    void convert_to(Dense *result) const override;

    void move_to(Dense *result) override;

protected:
    Dense(std::shared_ptr<const Executor> exec, size_type num_rows,
          size_type num_cols, size_type padding)
        : LinOp(exec, num_rows, num_cols, num_rows * padding),
          values_(exec, num_rows * padding),
          padding_(padding)
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
    size_type padding_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DENSE_HPP_
