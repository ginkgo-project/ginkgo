/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_FBCSR_HPP_
#define GKO_CORE_MATRIX_FBCSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/blockutils.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>

#include "matrix_strategies.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
class Dense;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;

template <typename ValueType, typename IndexType>
class Fbcsr;

template <typename ValueType, typename IndexType>
class FbcsrBuilder;


/**
 * FBCSR is a matrix format meant for matrices having a natural block structure
 * made up of small, dense, disjoint blocks. It is similar to CSR \sa Csr.
 * However, unlike Csr, each non-zero location stores a small dense block of
 * entries having a constant size. This reduces the number of integers that need
 * to be stored in order to refer to a given non-zero entry, and enables
 * efficient implementation of certain block methods.
 *
 * The block size is expected to be known in advance and passed to the
 * constructor.
 *
 * @note The total number of rows and the number of columns are expected to be
 *   divisible by the block size.
 *
 * The nonzero elements are stored in a 1D array row-wise, and accompanied
 * with a row pointer array which stores the starting index of each block-row.
 * An additional block-column index array is used to identify the block-column
 * of each nonzero block.
 *
 * The Fbcsr LinOp supports different operations:
 *
 * ```cpp
 * matrix::Fbcsr *A, *B, *C;      // matrices
 * matrix::Dense *b, *x;        // vectors tall-and-skinny matrices
 * matrix::Dense *alpha, *beta; // scalars of dimension 1x1
 * matrix::Identity *I;         // identity matrix
 *
 * // Applying to Dense matrices computes an SpMV/SpMM product
 * A->apply(b, x)              // x = A*b
 * A->apply(alpha, b, beta, x) // x = alpha*A*b + beta*x
 *
 * // Applying to Fbcsr matrices computes a SpGEMM product of two sparse
 * matrices A->apply(B, C)              // C = A*B A->apply(alpha, B, beta, C)
 * // C = alpha*A*B + beta*C
 *
 * // Applying to an Identity matrix computes a SpGEAM sparse matrix addition
 * A->apply(alpha, I, beta, B) // B = alpha*A + beta*B
 * ```
 * Both the SpGEMM and SpGEAM operation require the input matrices to be sorted
 * by block-column index, otherwise the algorithms will produce incorrect
 * results.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @ingroup fbcsr
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class Fbcsr : public EnableLinOp<Fbcsr<ValueType, IndexType>>,
              public EnableCreateMethod<Fbcsr<ValueType, IndexType>>,
              public ConvertibleTo<Fbcsr<next_precision<ValueType>, IndexType>>,
              public ConvertibleTo<Dense<ValueType>>,
              public ConvertibleTo<Csr<ValueType, IndexType>>,
              public ConvertibleTo<Coo<ValueType, IndexType>>,
              public ConvertibleTo<SparsityCsr<ValueType, IndexType>>,
              public DiagonalExtractable<ValueType>,
              public ReadableFromMatrixData<ValueType, IndexType>,
              public WritableToMatrixData<ValueType, IndexType>,
              public Transposable,
              public Permutable<IndexType>,
              public EnableAbsoluteComputation<
                  remove_complex<Fbcsr<ValueType, IndexType>>> {
    friend class EnableCreateMethod<Fbcsr>;
    friend class EnablePolymorphicObject<Fbcsr, LinOp>;
    friend class Coo<ValueType, IndexType>;
    friend class Dense<ValueType>;
    friend class SparsityCsr<ValueType, IndexType>;
    friend class FbcsrBuilder<ValueType, IndexType>;
    friend class Fbcsr<to_complex<ValueType>, IndexType>;

public:
    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = Fbcsr<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<Fbcsr>;

    using strategy_type =
        matrix_strategy::strategy_type<Fbcsr<value_type, index_type>>;


    void convert_to(Fbcsr<ValueType, IndexType> *result) const override;

    void move_to(Fbcsr<ValueType, IndexType> *result) override;

    friend class Fbcsr<next_precision<ValueType>, IndexType>;

    void convert_to(
        Fbcsr<next_precision<ValueType>, IndexType> *result) const override;

    void move_to(Fbcsr<next_precision<ValueType>, IndexType> *result) override;

    void convert_to(Dense<ValueType> *other) const override;

    void move_to(Dense<ValueType> *other) override;

    void convert_to(Csr<ValueType, IndexType> *result) const override;

    void move_to(Csr<ValueType, IndexType> *result) override;

    void convert_to(Coo<ValueType, IndexType> *result) const override;

    void move_to(Coo<ValueType, IndexType> *result) override;

    /// Get the block sparsity pattern in CSR-like format
    /** Note that the actual non-zero values are never copied;
     * the result always has a value array of size 1 with the value 1.
     */
    void convert_to(SparsityCsr<ValueType, IndexType> *result) const override;

    void move_to(SparsityCsr<ValueType, IndexType> *result) override;

    /// Convert COO data into block CSR
    /** @warning Unlike Csr::read, here explicit non-zeros are NOT dropped.
     */
    void read(const mat_data &data) override;

    void write(mat_data &data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    std::unique_ptr<LinOp> row_permute(
        const Array<IndexType> *permutation_indices) const override;

    std::unique_ptr<LinOp> column_permute(
        const Array<IndexType> *permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_row_permute(
        const Array<IndexType> *inverse_permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_column_permute(
        const Array<IndexType> *inverse_permutation_indices) const override;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    void compute_absolute_inplace() override;

    /**
     * Sorts all (value, col_idx) pairs in each row by column index
     */
    void sort_by_column_index();

    /*
     * Tests if all row entry pairs (value, col_idx) are sorted by column index
     *
     * @returns True if all row entry pairs (value, col_idx) are sorted by
     *          column index
     */
    bool is_sorted_by_column_index() const;

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type *get_values() noexcept { return values_.get_data(); }

    /// @see Fbcsr::get_const_values()
    const value_type *get_values() const noexcept
    {
        return values_.get_const_data();
    }

    /**
     * @copydoc Fbcsr::get_values()
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

    /// @see Fbcsr::get_const_col_idxs()
    const index_type *get_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * @copydoc Fbcsr::get_col_idxs()
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

    /// @see Fbcsr::get_const_row_ptrs()
    const index_type *get_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * @copydoc Fbcsr::get_row_ptrs()
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
     * Returns the starting rows.
     *
     * @return the starting rows.
     */
    index_type *get_srow() noexcept { return startrow_.get_data(); }

    const index_type *get_srow() const noexcept
    {
        return startrow_.get_const_data();
    }

    /**
     * @copydoc Fbcsr::get_srow()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_srow() const noexcept
    {
        return startrow_.get_const_data();
    }

    /**
     * Returns the number of the srow stored elements (involved warps)
     *
     * @return the number of the srow stored elements (involved warps)
     */
    size_type get_num_srow_elements() const noexcept
    {
        return startrow_.get_num_elems();
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

    /** Returns the strategy
     *
     * @return the strategy
     */
    std::shared_ptr<strategy_type> get_strategy() const noexcept
    {
        return strategy_;
    }

    /**
     * Set the strategy
     *
     * @param strategy the fbcsr strategy
     */
    void set_strategy(std::shared_ptr<strategy_type> strategy)
    {
        strategy_ = std::move(strategy->copy());
        this->make_srow();
    }

    int get_block_size() const { return bs_; }

    void set_block_size(const int block_size) { bs_ = block_size; }

    index_type get_num_block_rows() const
    {
        return row_ptrs_.get_num_elems() - 1;
    }

    index_type get_num_block_cols() const { return nbcols_; }

protected:
    using classical = matrix_strategy::classical<Fbcsr<value_type, index_type>>;

    /**
     * Creates an uninitialized FBCSR matrix with a block size of 1.
     *
     * @param exec  Executor associated to the matrix
     * @param strategy  the strategy of FBCSR
     */
    Fbcsr(std::shared_ptr<const Executor> exec,
          std::shared_ptr<strategy_type> strategy)
        : Fbcsr(std::move(exec), dim<2>{}, {}, 1, std::move(strategy))
    {}

    /**
     * Creates an uninitialized FBCSR matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param num_nonzeros  number of nonzeros
     * @param block_size Size of the small dense square blocks
     * @param strategy  the strategy of FBCSR
     */
    Fbcsr(std::shared_ptr<const Executor> exec, const dim<2> &size = dim<2>{},
          size_type num_nonzeros = {}, int block_size = 1,
          std::shared_ptr<strategy_type> strategy =
              std::make_shared<classical>());

    /**
     * Creates a FBCSR matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param block_size
     * @param values  array of matrix values
     * @param col_idxs  array of column indexes
     * @param row_ptrs  array of row pointers
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    Fbcsr(
        std::shared_ptr<const Executor> exec, const dim<2> &size,
        int block_size, ValuesArray &&values, ColIdxsArray &&col_idxs,
        RowPtrsArray &&row_ptrs,
        std::shared_ptr<strategy_type> strategy = std::make_shared<classical>())
        : EnableLinOp<Fbcsr>(exec, size),
          bs_{block_size},
          nbcols_{gko::blockutils::getNumBlocks(block_size, size[1])},
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)},
          startrow_(exec),
          strategy_(strategy->copy())
    {
        GKO_ASSERT_EQ(values_.get_num_elems(),
                      col_idxs_.get_num_elems() * bs_ * bs_);
        GKO_ASSERT_EQ(this->get_size()[0] / bs_ + 1, row_ptrs_.get_num_elems());
        this->make_srow();
    }

    void apply_impl(const LinOp *b, LinOp *x) const override;

    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const override;

    // TODO clean this up as soon as we improve strategy_type
    template <typename FbcsrType>
    void convert_strategy_helper(FbcsrType *result) const;

    /**
     * Computes srow. It should be run after changing any row_ptrs_ value.
     */
    void make_srow()
    {
        startrow_.resize_and_reset(
            strategy_->calc_size(col_idxs_.get_num_elems()));
        strategy_->process(row_ptrs_, &startrow_);
    }

private:
    int bs_;            ///< Block size
    size_type nbcols_;  ///< Number of block-columns
    Array<value_type> values_;
    Array<index_type> col_idxs_;
    Array<index_type> row_ptrs_;
    Array<index_type> startrow_;
    std::shared_ptr<strategy_type> strategy_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_FBCSR_HPP_
