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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/ell.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class BatchDense;

template <typename ValueType, typename IndexType>
class BatchEll;

typedef batch_stride batch_num_stored_elems_per_row;

/**
 * BatchEll is a matrix format which stores only the nonzero coefficients by
 * compressing each row of the matrix (compressed sparse row format). Each of
 * the individual batches are stored in a CSR matrix.
 *
 * @tparam ValueType  precision of matrix elements
 * @tparam IndexType  precision of matrix indexes
 *
 * @note Currently, BatchEll can store matrices with batch entries that have the
 * same sparsity pattern, but different values.
 *
 * @ingroup batch_ell
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class BatchEll
    : public EnableBatchLinOp<BatchEll<ValueType, IndexType>>,
      public EnableCreateMethod<BatchEll<ValueType, IndexType>>,
      public ConvertibleTo<BatchEll<next_precision<ValueType>, IndexType>>,
      public ConvertibleTo<BatchDense<ValueType>>,
      public BatchReadableFromMatrixData<ValueType, IndexType>,
      public BatchWritableToMatrixData<ValueType, IndexType>,
      public BatchTransposable,
      public BatchScaledIdentityAddable {
    friend class EnableCreateMethod<BatchEll>;
    friend class EnablePolymorphicObject<BatchEll, BatchLinOp>;
    friend class BatchEll<to_complex<ValueType>, IndexType>;

public:
    using BatchReadableFromMatrixData<ValueType, IndexType>::read;

    using value_type = ValueType;
    using index_type = IndexType;
    using transposed_type = BatchEll<ValueType, IndexType>;
    using unbatch_type = Ell<ValueType, IndexType>;
    using mat_data = matrix_data<ValueType, IndexType>;
    using absolute_type = remove_complex<BatchEll>;

    void convert_to(BatchEll<ValueType, IndexType>* result) const override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        result->values_ = this->values_;
        result->col_idxs_ = this->col_idxs_;
        result->stride_ = this->get_stride();
        result->num_stored_elems_per_row_ =
            this->get_num_stored_elements_per_row();
        result->set_size(this->get_size());
    }

    void move_to(BatchEll<ValueType, IndexType>* result) override
    {
        bool same_executor = this->get_executor() == result->get_executor();
        EnableBatchLinOp<BatchEll>::move_to(result);
    }
    friend class BatchEll<next_precision<ValueType>, IndexType>;

    void convert_to(
        BatchEll<next_precision<ValueType>, IndexType>* result) const override;

    void move_to(
        BatchEll<next_precision<ValueType>, IndexType>* result) override;

    void convert_to(BatchDense<ValueType>* result) const override;

    void move_to(BatchDense<ValueType>* result) override;

    void read(const std::vector<mat_data>& data) override;

    void write(std::vector<mat_data>& data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    static std::unique_ptr<BatchEll> create_from_batch_csc(
        std::shared_ptr<const Executor> exec, const size_type num_batch_entries,
        const dim<2>& size, const size_type num_elems_per_row,
        const gko::array<ValueType>& values,
        const gko::array<IndexType>& row_idxs,
        const gko::array<IndexType>& col_ptrs)
    {
        GKO_ASSERT_EQ(values.get_num_elems(),
                      row_idxs.get_num_elems() * num_batch_entries);
        GKO_ASSERT_EQ(size[1] + 1, col_ptrs.get_num_elems());

        auto batch_ell_mat = BatchEll<ValueType, IndexType>::create(
            exec, batch_dim<2>{num_batch_entries, size},
            batch_stride{num_batch_entries, num_elems_per_row});

        batch_ell_mat->create_from_batch_csc_impl(values, row_idxs, col_ptrs);
        return batch_ell_mat;
    }

    /**
     * Unbatches the BatchEll matrix into distinct matrices of Ell type.
     *
     * @return  a std::vector containing the distinct Ell matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const
    {
        auto exec = this->get_executor();
        auto unbatch_mats = std::vector<std::unique_ptr<unbatch_type>>{};
        size_type num_nnz =
            this->get_num_stored_elements() / this->get_num_batch_entries();
        size_type offset = 0;
        for (size_type b = 0; b < this->get_num_batch_entries(); ++b) {
            auto mat = unbatch_type::create(
                exec, this->get_size().at(b),
                this->get_num_stored_elements_per_row().at(b),
                this->get_stride().at(b));
            exec->copy_from(exec.get(), num_nnz,
                            this->get_const_values() + offset,
                            mat->get_values());
            exec->copy_from(exec.get(), num_nnz, this->get_const_col_idxs(),
                            mat->get_col_idxs());
            unbatch_mats.emplace_back(std::move(mat));
            offset += num_nnz;
        }
        return unbatch_mats;
    }

    /**
     * Returns the values of the matrix.
     *
     * @return the values of the matrix.
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * @copydoc BatchEll::get_values()
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
     * @copydoc BatchEll::get_col_idxs()
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
     * Returns the number of stored elements per row.
     *
     * @return the number of stored elements per row.
     */
    const batch_num_stored_elems_per_row& get_num_stored_elements_per_row()
        const noexcept
    {
        return num_stored_elems_per_row_;
    }

    /**
     * Returns the number of elements explicitly stored in the matrix,
     * cumulative over all the batches
     *
     * @return the number of elements explicitly stored in the matrix,
     * cumulative over all the batches
     */
    const batch_stride& get_stride() const noexcept { return stride_; }

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
     * Returns the `idx`-th non-zero element of the `row`-th row .
     *
     * @param row  the row of the requested element
     * @param idx  the idx-th stored element of the row
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& val_at(size_type id, size_type row, size_type idx) noexcept
    {
        return values_.get_data()[this->linearize_index(id, row, idx)];
    }

    /**
     * @copydoc Ell::val_at(size_type, size_type)
     */
    value_type val_at(size_type id, size_type row, size_type idx) const noexcept
    {
        return values_.get_const_data()[this->linearize_index(id, row, idx)];
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
    index_type& col_at(size_type row, size_type idx) noexcept
    {
        return this->get_col_idxs()[this->linearize_index(0, row, idx)];
    }

    /**
     * @copydoc Ell::col_at(size_type, size_type)
     */
    index_type col_at(size_type row, size_type idx) const noexcept
    {
        return this->get_const_col_idxs()[this->linearize_index(0, row, idx)];
    }

    /**
     * Creates an immutable BatchEll matrix from a set of constant arrays.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param col_idxs  the column index array of the matrix
     * @param num_stored_elems_per_row  the number of stored nonzeros per row
     * @param stride  the column-stride of the value and column index array
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if they reside on the same executor as the matrix) or a copy of
     *          the arrays on the correct executor.
     */
    static std::unique_ptr<const BatchEll> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
        const batch_stride& num_stored_elems_per_row,
        const batch_stride& stride,
        gko::detail::const_array_view<ValueType>&& values,
        gko::detail::const_array_view<IndexType>&& col_idxs)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const BatchEll>(
            new BatchEll{exec, size, num_stored_elems_per_row, stride,
                         gko::detail::array_const_cast(std::move(values)),
                         gko::detail::array_const_cast(std::move(col_idxs))});
    }

protected:
    /**
     * Creates an uninitialized BatchEll matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param size  the common size of all the batch matrices
     * @param num_nonzeros  number of nonzeros in each of the batch matrices
     */
    BatchEll(std::shared_ptr<const Executor> exec,
             const batch_dim<2>& size = batch_dim<2>{})
        : EnableBatchLinOp<BatchEll>(exec, size),
          num_stored_elems_per_row_(batch_num_stored_elems_per_row{}),
          values_(exec),
          col_idxs_(exec)
    {}

    /**
     * Creates an uninitialized BatchEll matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param size  the common size of all the batch matrices
     * @param num_nonzeros  number of nonzeros in each of the batch matrices
     */
    BatchEll(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             const batch_stride& num_stored_elems_per_row)
        : EnableBatchLinOp<BatchEll>(exec, size),
          num_stored_elems_per_row_(num_stored_elems_per_row),
          stride_(get_stride_from_dim(size)),
          values_(exec, num_stored_elems_per_row.at(0) * size.at(0)[0] *
                            size.get_num_batch_entries()),
          col_idxs_(exec, num_stored_elems_per_row.at(0) * size.at(0)[0])
    {
        GKO_ASSERT_EQUAL_BATCH_ENTRIES(size, num_stored_elems_per_row);
    }

    /**
     * Creates an uninitialized BatchEll matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param size  the common size of all the batch matrices
     * @param num_nonzeros  number of nonzeros in each of the batch matrices
     */
    BatchEll(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             const batch_stride& num_stored_elems_per_row,
             const batch_stride& stride)
        : EnableBatchLinOp<BatchEll>(exec, size),
          num_stored_elems_per_row_(num_stored_elems_per_row),
          stride_(stride),
          values_(exec, num_stored_elems_per_row.at(0) * stride.at(0) *
                            size.get_num_batch_entries()),
          col_idxs_(exec, num_stored_elems_per_row.at(0) * stride.at(0))
    {
        GKO_ASSERT_EQUAL_BATCH_ENTRIES(size, num_stored_elems_per_row);
        GKO_ASSERT_EQUAL_BATCH_ENTRIES(size, stride);
    }

    /**
     * Creates an BatchEll matrix by duplicating a CSR matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_batch_entries  the number of batches to be stored
     * @param csr_mat  the csr matrix to be duplicated
     */
    BatchEll(std::shared_ptr<const Executor> exec,
             const size_type num_batch_entries,
             const matrix::Ell<value_type, index_type>* ell_mat)
        : EnableBatchLinOp<BatchEll>(
              exec, batch_dim<2>(num_batch_entries, ell_mat->get_size())),
          num_stored_elems_per_row_(batch_num_stored_elems_per_row(
              num_batch_entries, ell_mat->get_num_stored_elements_per_row())),
          stride_(batch_stride(num_batch_entries, ell_mat->get_stride())),
          values_(exec, ell_mat->get_num_stored_elements_per_row() *
                            ell_mat->get_stride() * num_batch_entries),
          col_idxs_(exec, ell_mat->get_num_stored_elements_per_row() *
                              ell_mat->get_stride())
    {
        batch_duplicator(exec, num_batch_entries, col_idxs_.get_num_elems(),
                         ell_mat, this);
    }

    /**
     * Creates an BatchEll matrix by duplicating a BatchEll matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  the number of times the entire input batch
     *                          matrix has to be duplicated
     * @param batch_mat  the BatchEll matrix to be duplicated
     */
    BatchEll(std::shared_ptr<const Executor> exec,
             const size_type num_duplications,
             const matrix::BatchEll<value_type, index_type>* batch_mat)
        : EnableBatchLinOp<BatchEll>(
              exec, batch_dim<2>(
                        num_duplications * batch_mat->get_num_batch_entries(),
                        batch_mat->get_size().at(0))),
          num_stored_elems_per_row_(
              batch_mat->get_num_stored_elements_per_row()),
          stride_(batch_mat->get_stride()),
          values_(exec,
                  batch_mat->get_num_stored_elements() * num_duplications),
          col_idxs_(exec, batch_mat->get_num_stored_elements() /
                              batch_mat->get_num_batch_entries())
    {
        batch_duplicator(exec, num_duplications, col_idxs_.get_num_elems(),
                         batch_mat, this);
    }


    /**
     * Creates a BatchEll matrix from already allocated (and initialized) row
     * pointer, column index and value arrays.
     *
     * @tparam ValuesArray  type of `values` array
     * @tparam ColIdxsArray  type of `col_idxs` array
     *
     * @param exec  Executor associated to the matrix
     * @param size  the batch size of the batch matrices
     * @param num_stored_elems_per_row  The (common) number of stored entries
     *                                  per row.
     * @param stride  (Common) row-stride of the small matrices.
     * @param values  array of matrix values concatenated for the different
     *                batches
     * @param col_idxs  array of column indexes which is common among all the
     *                  batches
     *
     * @note If one of `col_idxs` or `values` is not an rvalue, not
     *       an array of IndexType, IndexType and ValueType, respectively, or
     *       is on the wrong executor, an internal copy of that array will be
     *       created, and the original array data will not be used in the
     *       matrix.
     */
    template <typename ValuesArray, typename ColIdxsArray>
    BatchEll(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
             const batch_stride& num_stored_elems_per_row,
             const batch_stride& stride, ValuesArray&& values,
             ColIdxsArray&& col_idxs)
        : EnableBatchLinOp<BatchEll>(exec, size.get_num_batch_entries(),
                                     size.at(0)),
          num_stored_elems_per_row_(num_stored_elems_per_row),
          stride_(stride),
          values_{exec, std::forward<ValuesArray>(values)},
          col_idxs_{exec, std::forward<ColIdxsArray>(col_idxs)}
    {
        GKO_ASSERT_EQ(values_.get_num_elems(),
                      col_idxs_.get_num_elems() * size.get_num_batch_entries());
    }

    void create_from_batch_csc_impl(const gko::array<ValueType>& values,
                                    const gko::array<IndexType>& row_idxs,
                                    const gko::array<IndexType>& col_ptrs);

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

private:
    template <typename MatrixType>
    void batch_duplicator(std::shared_ptr<const Executor> exec,
                          const size_type num_batch_entries,
                          const size_type col_idxs_size,
                          const MatrixType* input,
                          BatchEll<value_type, index_type>* output)
    {
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
    }

    batch_num_stored_elems_per_row num_stored_elems_per_row_;
    batch_stride stride_;
    array<value_type> values_;
    array<index_type> col_idxs_;
    array<index_type> row_ptrs_;

    size_type linearize_index(size_type batch_id, size_type row,
                              size_type col) const noexcept
    {
        size_type num_elems_per_batch =
            values_.get_num_elems() / this->get_num_batch_entries();
        return batch_id * num_elems_per_batch + row +
               stride_.at(batch_id) * col;
    }

    batch_stride get_stride_from_dim(const batch_dim<2>& dim)
    {
        if (dim.stores_equal_sizes()) {
            return batch_stride(dim.get_num_batch_entries(), dim.at(0)[0]);
        } else {
            std::vector<size_type> strides;
            strides.reserve(dim.get_num_batch_entries());
            auto batch_sizes = dim.get_batch_sizes();
            for (auto dims : batch_sizes) {
                strides.push_back(dims[0]);
            }
            return batch_stride(strides);
        }
    }

    void add_scaled_identity_impl(const BatchLinOp* a,
                                  const BatchLinOp* b) override;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_ELL_HPP_
