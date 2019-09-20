/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_SPARSITY_HPP_
#define GKO_CORE_MATRIX_SPARSITY_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace matrix {


/**
 * SPARSITY is a matrix format which stores only the sparsity pattern of a
 * sparse matrix by compressing each row of the matrix (compressed sparse row
 * format).
 *
 * The values of the nonzero elements are stored a value array of length 1. All
 * the values in the matrix are equal to this value. By default, this value is
 * set to 1.0. A row pointer array which stores the starting index of each row.
 * An additional column index array is used to identify the column where a
 * nonzero is present.
 *
 * @tparam ValueType  precision of vectors in apply
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup sparsity
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Sparsity : public EnableLinOp<Sparsity<ValueType, IndexType>>,
                 public EnableCreateMethod<Sparsity<ValueType, IndexType>>,
                 public ReadableFromMatrixData<ValueType, IndexType>,
                 public WritableToMatrixData<ValueType, IndexType>,
                 public Transposable {
    friend class EnableCreateMethod<Sparsity>;
    friend class EnablePolymorphicObject<Sparsity, LinOp>;

public:
    using EnableLinOp<Sparsity>::convert_to;
    using EnableLinOp<Sparsity>::move_to;

    using index_type = IndexType;
    using value_type = ValueType;

    using mat_data = matrix_data<ValueType, IndexType>;

    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Transforms the sparsity matrix to an adjacency matrix. As the adjacency
     * matrix has to be square, the input Sparsity matrix for this function to
     * work has to be square.
     *
     * Note: The adjacency matrix in this case is the sparsity pattern but with
     * the diagonal ones removed. This is mainly used for the
     * reordering/partitioning as taken in by graph libraries such as METIS.
     */
    std::unique_ptr<Sparsity> to_adjacency_matrix() const;

    /**
     * Sorts each row by column index
     */
    void sort_by_column_index();

    /*
     * Tests if all col_idxs are sorted by column index
     *
     * @returns True if all col_idxs are sorted.
     */
    bool is_sorted_by_column_index() const;

    /**
     * Returns the column indices of the matrix.
     *
     * @return the column indices of the matrix.
     */
    index_type *get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc Sparsity::get_col_idxs()
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
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type *get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc Sparsity::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * Returns the value stored in the matrix.
     *
     * @return the value of the matrix.
     */
    value_type *get_value() noexcept { return value_.get_data(); }

    /**
     * @copydoc Sparsity::get_value()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type *get_const_value() const noexcept
    {
        return value_.get_const_data();
    }


    /**
     * Returns the number of elements explicitly stored in the matrix.
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_nonzeros() const noexcept
    {
        return col_idxs_.get_num_elems();
    }

protected:
    /**
     * Creates an uninitialized SPARSITY matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     */
    Sparsity(std::shared_ptr<const Executor> exec,
             const dim<2> &size = dim<2>{}, size_type num_nonzeros = {})
        : EnableLinOp<Sparsity>(exec, size),
          col_idxs_(exec, num_nonzeros),
          // avoid allocation for empty matrix
          row_ptrs_(exec, size[0] + (size[0] > 0)),
          value_(exec, 1)
    {}

    /**
     * Creates a SPARSITY matrix from already allocated (and initialized) row
     * pointer, column index arrays.
     *
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     * @param value  value stored. (same value for all matrix elements)
     *
     * @note If one of `row_ptrs` or `col_idxs` is not an rvalue, not
     *       an array of IndexType and IndexType respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ColIdxsArray, typename RowPtrsArray>
    Sparsity(std::shared_ptr<const Executor> exec, const dim<2> &size,
             ColIdxsArray &&col_idxs, RowPtrsArray &&row_ptrs,
             value_type value = 1.0)
        : EnableLinOp<Sparsity>(exec, size),
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)}
    {
        value_type val = value;
        auto tmp = Array<value_type>{exec->get_master(), 1};
        tmp.get_data()[0] = val;
        value_ = Array<value_type>{exec, std::move(tmp)};
        GKO_ENSURE_IN_BOUNDS(col_idxs_.get_num_elems() - 1,
                             col_idxs_.get_num_elems());
        GKO_ENSURE_IN_BOUNDS(this->get_size()[0], row_ptrs_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

private:
    Array<index_type> col_idxs_;
    Array<index_type> row_ptrs_;
    Array<value_type> value_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SPARSITY_HPP_
