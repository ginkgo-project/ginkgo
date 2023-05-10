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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class BatchDense;

template <typename ValueType, typename IndexType>
class BatchCsr;


/**
 * BatchCsr is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format). Each of
 * the individual batches are stored in a CSR matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @note Currently, BatchCsr can store matrices with batch entries that have the
 * same sparsity pattern, but different values.
 *
 * @ingroup batch_csr
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchCsr
    : public EnableBatchLinOp<BatchCsr<ValueType, IndexType>>,
      public EnableCreateMethod<BatchCsr<ValueType, IndexType>>,
      public ConvertibleTo<BatchCsr<next_precision<ValueType>, IndexType>>,
      public ConvertibleTo<BatchDense<ValueType>>,
      public BatchReadableFromMatrixData<ValueType, IndexType>,
      public BatchWritableToMatrixData<ValueType, IndexType>,
      public BatchTransposable,
      public BatchScaledIdentityAddable {
    friend class EnableCreateMethod<BatchCsr>;
    friend class EnablePolymorphicObject<BatchCsr, BatchLinOp>;
    friend class BatchCsr<to_complex<ValueType>, IndexType>;

public:
    using BatchReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = BatchCsr<ValueType, IndexType>;
    using unbatch_type = Csr<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<BatchCsr>;

    void convert_to(BatchCsr<ValueType, IndexType>* result) const override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        result->values_ = this->values_;
        result->col_idxs_ = this->col_idxs_;
        result->row_ptrs_ = this->row_ptrs_;
        result->set_size(this->get_size());
    }

    void move_to(BatchCsr<ValueType, IndexType>* result) override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        EnableBatchLinOp<BatchCsr>::move_to(result);
    }
    friend class BatchCsr<next_precision<ValueType>, IndexType>;

    void convert_to(
        BatchCsr<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(
        BatchCsr<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(BatchDense<ValueType>* result) const override;

    void move_to(BatchDense<ValueType>* result) override;

    void read(const std::vector<mat_data>& data) override;

    void write(std::vector<mat_data>& data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Unbatches the BatchCsr matrix into distinct matrices of Csr type.
     *
     * @return  a std::vector containing the distinct Csr matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const
    {
        auto exec = this->get_executor();
        auto unbatch_mats = std::vector<std::unique_ptr<unbatch_type>>{};
        size_type num_nnz =
            this->get_num_stored_elements() / this->get_num_batch_entries();
        size_type offset = 0;
        for (size_type b = 0; b < this->get_num_batch_entries(); ++b) {
            auto mat =
                unbatch_type::create(exec, this->get_size().at(b), num_nnz);
            exec->copy_from(exec.get(), num_nnz,
                            this->get_const_values() + offset,
                            mat->get_values());
            exec->copy_from(exec.get(), num_nnz, this->get_const_col_idxs(),
                            mat->get_col_idxs());
            exec->copy_from(exec.get(), this->get_size().at(b)[0] + 1,
                            this->get_const_row_ptrs(), mat->get_row_ptrs());
            unbatch_mats.emplace_back(std::move(mat));
            offset += num_nnz;
        }
        return unbatch_mats;
    }

    /**
     * Sorts all (value, col_idx) pairs in each row by column index
     */
    void sort_by_column_index();

    /**
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
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc BatchCsr::get_values()
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
     * Returns the column indexes of the matrix.
     *
     * @return the column indexes of the matrix.
     */
    index_type* get_col_idxs() noexcept { return col_idxs_.get_data(); }

    /**
     * @copydoc BatchCsr::get_col_idxs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_col_idxs() const noexcept
    {
        return col_idxs_.get_const_data();
    }

    /**
     * Returns the row pointers of the matrix.
     *
     * @return the row pointers of the matrix.
     */
    index_type* get_row_ptrs() noexcept { return row_ptrs_.get_data(); }

    /**
     * @copydoc BatchCsr::get_row_ptrs()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_ptrs() const noexcept
    {
        return row_ptrs_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the matrix,
     * cumulative over all the batches
     *
     * @return the number of elements explicitly stored in the matrix,
     * cumulative over all the batches
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Creates a constant BatchCsr matrix by wrapping a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param row_ptrs  the row pointer array of the matrix
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          these arrays on the correct executor.
     */
    static std::unique_ptr<const BatchCsr> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs,
        gko::detail::const_array_view<IndexType>&& row_ptrs)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const BatchCsr>(new BatchCsr{
            exec, size, gko::detail::array_const_cast(std::move(values)),
            gko::detail::array_const_cast(std::move(col_idxs)),
            gko::detail::array_const_cast(std::move(row_ptrs))});
    }

protected:
    /**
     * Creates an uninitialized BatchCsr matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  an object describing the size of the batch matrices.
     * @param num_nonzeros  number of nonzeros in each of the batch matrices
     */
    BatchCsr(std::shared_ptr<const Executor> exec,
             const batch_dim<2>& size = batch_dim<2>{},
             size_type num_nonzeros = {})
        : EnableBatchLinOp<BatchCsr>(exec, size),
          values_(exec, num_nonzeros * size.get_num_batch_entries()),
          col_idxs_(exec, num_nonzeros),
          row_ptrs_(exec, (size.at(0)[0]) + 1)
    {
        if (!size.stores_equal_sizes()) {
            GKO_NOT_IMPLEMENTED;
        }
    }


    /**
     * Creates an uninitialized BatchCsr matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param size  the common size of all the batch matrices
     * @param num_nonzeros  number of nonzeros in each of the batch matrices
     */
    BatchCsr(std::shared_ptr<const Executor> exec,
             const size_type num_batch_entries, const dim<2>& size,
             size_type num_nonzeros)
        : BatchCsr(exec, batch_dim<>{num_batch_entries, size}, num_nonzeros)
    {}


    /**
     * Creates an BatchCsr matrix by duplicating a CSR matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param csr_mat  the csr matrix to be duplicated
     */
    BatchCsr(std::shared_ptr<const Executor> exec,
             const size_type num_batch_entries,
             const matrix::Csr<value_type, index_type>* csr_mat)
        : EnableBatchLinOp<BatchCsr>(
              exec, batch_dim<2>(num_batch_entries, csr_mat->get_size())),
          values_(exec, csr_mat->get_num_stored_elements() * num_batch_entries),
          col_idxs_(exec, csr_mat->get_num_stored_elements()),
          row_ptrs_(exec, (csr_mat->get_size()[0]) + 1)
    {
        batch_duplicator(exec, num_batch_entries,
                         csr_mat->get_num_stored_elements(), csr_mat, this);
    }

    /**
     * Creates an BatchCsr matrix by duplicating a BatchCsr matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  the number of times the entire input batch
     *                          matrix has to be duplicated
     * @param batch_mat  the BatchCsr matrix to be duplicated
     */
    BatchCsr(std::shared_ptr<const Executor> exec,
             const size_type num_duplications,
             const matrix::BatchCsr<value_type, index_type>* batch_mat)
        : EnableBatchLinOp<BatchCsr>(
              exec, batch_dim<2>(
                        num_duplications * batch_mat->get_num_batch_entries(),
                        batch_mat->get_size().at(0))),
          values_(exec,
                  batch_mat->get_num_stored_elements() * num_duplications),
          col_idxs_(exec, batch_mat->get_num_stored_elements() /
                              batch_mat->get_num_batch_entries()),
          row_ptrs_(exec, (batch_mat->get_size().at(0)[0]) + 1)
    {
        batch_duplicator(exec, num_duplications, col_idxs_.get_num_elems(),
                         batch_mat, this);
    }


    /**
     * Creates a BatchCsr matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  the batch size of the batch matrices
     * @param values  array of matrix values concatenated for the different
     *                batches
     * @param col_idxs  array of column indexes which is common among all the
     *                  batches
     * @param row_ptrs  array of row pointers which is common among all the
     *                  batches
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    BatchCsr(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             ValuesArray&& values, ColIdxsArray&& col_idxs,
             RowPtrsArray&& row_ptrs)
        : EnableBatchLinOp<BatchCsr>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)},
          row_ptrs_{exec, std::forward<RowPtrsArray>(row_ptrs)}
    {
        GKO_ASSERT_EQ(values_.get_num_elems(),
                      col_idxs_.get_num_elems() * size.get_num_batch_entries());
        GKO_ASSERT_EQ(this->get_size().at(0)[0] + 1, row_ptrs_.get_num_elems());
    }


    /**
     * Creates a BatchCsr matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     * @tparam RowPtrsArray  type of `row_ptrs` array
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches
     * @param size  the common size of the batch matrices
     * @param values  array of matrix values concatenated for the different
     *                batches
     * @param col_idxs  array of column indexes which is common among all the
     *                  batches
     * @param row_ptrs  array of row pointers which is common among all the
     *                  batches
     *
     * @note If one of `row_ptrs`, `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray,
              typename RowPtrsArray>
    BatchCsr(std::shared_ptr<const Executor> exec,
             const size_type num_batch_entries, const dim<2>& size,
             ValuesArray&& values, ColIdxsArray&& col_idxs,
             RowPtrsArray&& row_ptrs)
        : BatchCsr(exec, batch_dim<2>(num_batch_entries, size),
                   std::forward<ValuesArray>(values),
                   std::forward<ColIdxsArray>(col_idxs),
                   std::forward<RowPtrsArray>(row_ptrs))
    {}

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

private:
    template <typename MatrixType>
    void batch_duplicator(std::shared_ptr<const Executor> exec,
                          const size_type num_batch_entries,
                          const size_type col_idxs_size,
                          const MatrixType* input,
                          BatchCsr<value_type, index_type>* output)
    {
        auto row_ptrs = output->get_row_ptrs();
        auto col_idxs = output->get_col_idxs();
        auto values = output->get_values();
        size_type offset = 0;
        size_type num_nnz = input->get_num_stored_elements();
        for (size_type i = 0; i < num_batch_entries; ++i) {
            exec->copy_from(input->get_executor().get(), num_nnz,
                            input->get_const_values(), values + offset);
            offset += num_nnz;
        }
        exec->copy_from(input->get_executor().get(), col_idxs_size,
                        input->get_const_col_idxs(), col_idxs);
        exec->copy_from(input->get_executor().get(),
                        output->get_size().at(0)[0] + 1,
                        input->get_const_row_ptrs(), row_ptrs);
    }

    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;

    void add_scaled_identity_impl(const BatchLinOp* a,
                                  const BatchLinOp* b) override;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_CSR_HPP_
