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
#include "core/base/lin_op.hpp"
#include "core/base/mtx_reader.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;


/**
 * ELL is a matrix format where stride with explicit zeros is used such that
 * all rows have the same number of stored elements. The number of elements
 * stored in each row is the largest number of nonzero elements in any of the
 * rows (obtainable through get_max_nonzeros_per_row() method). This removes
 * the need of a row pointer like in the CSR format, and allows for SIMD
 * processing of the distinct rows. For efficient processing, the nonzero
 * elements and the corresponding column indices are stored in column-major
 * fashion. The columns are padded to the length by user-defined stride
 * parameter whose default value is the number of rows of the matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Ell : public EnableLinOp<Ell<ValueType, IndexType>>,
            public EnableCreateMethod<Ell<ValueType, IndexType>>,
            public ConvertibleTo<Dense<ValueType>>,
            public ReadableFromMatrixData<ValueType, IndexType>,
            public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableCreateMethod<Ell>;
    friend class EnablePolymorphicObject<Ell, LinOp>;
    friend class Dense<ValueType>;

public:
    using EnableLinOp<Ell>::convert_to;
    using EnableLinOp<Ell>::move_to;

    using value_type = ValueType;
    using index_type = IndexType;
    using mat_data = matrix_data<ValueType, IndexType>;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

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
     * Returns the maximum number of non-zeros per row.
     *
     * @return the maximum number of non-zeros per row.
     */
    size_type get_max_nonzeros_per_row() const noexcept
    {
        return max_nonzeros_per_row_;
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
     * Returns the `idx`-th non-zero element of the `row`-th row .
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type &val_at(size_type row, size_type idx) noexcept
    {
        return values_.get_data()[this->linearize_index(row, idx)];
    }

    /**
     * @copydoc Ell::val_at(size_type, size_type)
     */
    value_type val_at(size_type row, size_type idx) const noexcept
    {
        return values_.get_const_data()[this->linearize_index(row, idx)];
    }

    /**
     * Returns the `idx`-th column index of the `row`-th row .
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    index_type &col_at(size_type row, size_type idx) noexcept
    {
        return this->get_col_idxs()[this->linearize_index(row, idx)];
    }

    /**
     * @copydoc Ell::col_at(size_type, size_type)
     */
    index_type col_at(size_type row, size_type idx) const noexcept
    {
        return this->get_const_col_idxs()[this->linearize_index(row, idx)];
    }

protected:
    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.
     *     The max_nonzeros_per_row is set to the number of cols of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Ell(std::shared_ptr<const Executor> exec, const dim &size = dim{})
        : Ell(std::move(exec), size, size.num_cols)
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *    (The stride is set to the number of rows of the matrix.)
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param max_nonzeros_per_row   maximum number of nonzeros in one row
     */
    Ell(std::shared_ptr<const Executor> exec, const dim &size,
        size_type max_nonzeros_per_row)
        : Ell(std::move(exec), size, max_nonzeros_per_row, size.num_rows)
    {}

    /**
     * Creates an uninitialized Ell matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param max_nonzeros_per_row   maximum number of nonzeros in one row
     * @param stride                stride of the rows
     */
    Ell(std::shared_ptr<const Executor> exec, const dim &size,
        size_type max_nonzeros_per_row, size_type stride)
        : EnableLinOp<Ell>(exec, size),
          values_(exec, stride * max_nonzeros_per_row),
          col_idxs_(exec, stride * max_nonzeros_per_row),
          max_nonzeros_per_row_(max_nonzeros_per_row),
          stride_(stride)
    {}


    /**
     * Creates an ELL matrix from already allocated (and initialized)
     * column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param max_nonzeros_per_row   maximum number of nonzeros in one row
     * @param stride  stride of the rows
     *
     * @note If one of `col_idxs` or `values` is not an rvalue, not an array of
     *       IndexType and ValueType, respectively, or is on the wrong executor,
     *       an internal copy of that array will be created, and the original
     *       array data will not be used in the matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray>
    Ell(std::shared_ptr<const Executor> exec, const dim &size,
        ValuesArray &&values, ColIdxsArray &&col_idxs,
        size_type max_nonzeros_per_row, size_type stride)
        : EnableLinOp<Ell>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          max_nonzeros_per_row_{max_nonzeros_per_row},
          stride_{stride}
    {
        ENSURE_IN_BOUNDS(max_nonzeros_per_row_ * stride_ - 1,
                         values_.get_num_elems());
        ENSURE_IN_BOUNDS(max_nonzeros_per_row_ * stride_ - 1,
                         col_idxs_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row + stride_ * col;
    }

private:
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    size_type max_nonzeros_per_row_;
    size_type stride_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_ELL_HPP_
