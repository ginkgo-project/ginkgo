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

#ifndef GKO_CORE_MATRIX_ELL_HPP_
#define GKO_CORE_MATRIX_ELL_HPP_


#include "core/base/array.hpp"
#include "core/base/convertible.hpp"
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"


namespace gko {
namespace matrix {

template <typename ValueType>
class Dense;

/**
 * Ell is a matrix format which aligned all nonzero elements to the left
 * of the matrix (from the first column) and save the smaller matrix up
 * to the maximum number of nonzero elements in a single element.
 *
 * The nonzero elements are stored in a 1D array column-wise, and accompanied
 * with a column index array index of each row.
 * An additional value max_nnz_row is also stored to identify the size.
 * (Column) padding is also introduced to improve the performance.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ell : public BasicLinOp<Ell<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ReadableFromMtx {
    friend class BasicLinOp<Ell>;
    friend class Dense<ValueType>;

public:
    using BasicLinOp<Ell>::create;
    using BasicLinOp<Ell>::convert_to;
    using BasicLinOp<Ell>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;

    void apply(const LinOp *b, LinOp *x) const override;

    void apply(const LinOp *alpha, const LinOp *b, const LinOp *beta,
               LinOp *x) const override;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read_from_mtx(const std::string &filename) override;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc Ell::get_values()
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
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Ell::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the maximum number of one row.
     */
    size_type get_max_nnz_row() const noexcept { return max_nnz_row_; }

    /**
     * Returns the padding of the matrix.
     */
    size_type get_padding() const noexcept { return padding_; }

    /**
     * Returns a value in the ELLPACK format.
     *
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    value_type &val_at(size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    /**
     * @copydoc Ell::val_at(size_type, size_type)
     */
    value_type val_at(size_type row, size_type col) const noexcept
    {
        return values_.get_const_data()[linearize_index(row, col)];
    }

    /**
     * Returns a column index of the ELLPACK foramt.
     *
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the CPU results in a runtime error)
     */
    index_type &col_at(size_type row, size_type col) noexcept
    {
        return get_col_idxs()[linearize_index(row, col)];
    }

    /**
     * @copydoc Ell::col_at(size_type, size_type)
     */
    index_type col_at(size_type row, size_type col) const noexcept
    {
        return get_const_col_idxs()[linearize_index(row, col)];
    }

protected:
    /**
     * Creates an empty ELL matrix.
     *
     * @param exec  Executor associated to the matrix
     */
    explicit Ell(std::shared_ptr<const Executor> exec)
        : BasicLinOp<Ell>(exec, 0, 0, 0),
          values_(exec),
          col_idxs_(exec),
          max_nnz_row_(0)
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows      number of rows
     * @param num_cols      number of columns
     * @param num_nonzeros  number of nonzeros
     * @param max_nnz_row   maximum number of nonzeros in one row
     * @param padding       padding of the rows
     */
    Ell(std::shared_ptr<const Executor> exec, size_type num_rows,
        size_type num_cols, size_type num_nonzeros, size_type max_nnz_row,
        size_type padding)
        : BasicLinOp<Ell>(exec, num_rows, num_cols, num_nonzeros),
          values_(exec, num_rows*max_nnz_row),
          col_idxs_(exec, num_rows*max_nnz_row),
          max_nnz_row_(max_nnz_row),
          padding_(padding)
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_rows      number of rows
     * @param num_cols      number of columns
     * @param num_nonzeros  number of nonzeros
     * @param max_nnz_row   maximum number of nonzeros in one row
     */
    Ell(std::shared_ptr<const Executor> exec, size_type num_rows,
        size_type num_cols, size_type num_nonzeros, size_type max_nnz_row)
        : Ell(std::move(exec), num_rows, num_cols, num_nonzeros,
              max_nnz_row, num_rows)
    {}

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row + padding_ * col;
    }

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    size_type max_nnz_row_;
    size_type padding_;

};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_ELL_HPP_
