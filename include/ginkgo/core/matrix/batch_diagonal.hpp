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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class BatchDense;


/**
 * BatchDiagonal is a batch matrix format which explicitly stores all values of
 * the matrix in each of the batches.
 *
 * The values in each of the batches are stored in row-major format (values
 * belonging to the same row appear consecutive in the memory). Optionally, rows
 * can be padded for better memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 * @ingroup batch_dense
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchDiagonal
    : public EnableBatchLinOp<BatchDiagonal<ValueType>>,
      public EnableCreateMethod<BatchDiagonal<ValueType>>,
      public ConvertibleTo<BatchDiagonal<next_precision<ValueType>>>,
      public BatchReadableFromMatrixData<ValueType, int32>,
      public BatchReadableFromMatrixData<ValueType, int64>,
      public BatchWritableToMatrixData<ValueType, int32>,
      public BatchWritableToMatrixData<ValueType, int64>,
      public BatchTransposable {
    friend class EnableCreateMethod<BatchDiagonal>;
    friend class EnablePolymorphicObject<BatchDiagonal, BatchLinOp>;
    friend class BatchDiagonal<to_complex<ValueType>>;

public:
    using EnableBatchLinOp<BatchDiagonal>::convert_to;
    using EnableBatchLinOp<BatchDiagonal>::move_to;
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = BatchDiagonal<ValueType>;
    using unbatch_type = Diagonal<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchDiagonal>;
    using complex_type = to_complex<BatchDiagonal>;

    /**
     * Creates a BatchDiagonal matrix with the configuration of another
     * BatchDiagonal matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchDiagonal> create_with_config_of(
        const BatchDiagonal* other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    friend class BatchDiagonal<next_precision<ValueType>>;

    void convert_to(
        BatchDiagonal<next_precision<ValueType>>* result) const override;

    void move_to(BatchDiagonal<next_precision<ValueType>>* result) override;

    /**
     * Read from a COO-type matrix data object into this batch diagonal matrix.
     *
     * Any off-diagonal entries in the input are ignored.
     */
    void read(const std::vector<mat_data>& data) override;

    /**
     * Read from a COO-type matrix data object into this batch diagonal matrix.
     *
     * Any off-diagonal entries in the input are ignored.
     */
    void read(const std::vector<mat_data32>& data) override;

    void write(std::vector<mat_data>& data) const override;

    void write(std::vector<mat_data32>& data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Unbatches the batched dense and creates a std::vector of Diagonal
     * matrices
     *
     * @return  a std::vector containing the Diagonal matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);
        auto exec = this->get_executor();
        auto unbatch_mats = std::vector<std::unique_ptr<unbatch_type>>{};
        const auto c_entry_size =
            std::min(this->get_size().at(0)[0], this->get_size().at(0)[1]);
        for (size_type b = 0; b < this->get_num_batch_entries(); ++b) {
            auto mat = unbatch_type::create(exec, this->get_size().at(b)[0]);
            if (this->get_size().stores_equal_sizes()) {
                exec->copy_from(exec.get(), mat->get_size()[0],
                                this->get_const_values() + b * c_entry_size,
                                mat->get_values());
            } else {
                exec->copy_from(
                    exec.get(), mat->get_size()[0],
                    this->get_const_values() +
                        num_elems_per_batch_cumul_.get_const_data()[b],
                    mat->get_values());
            }
            unbatch_mats.emplace_back(std::move(mat));
        }
        return unbatch_mats;
    }

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values() noexcept { return values_.get_data(); }

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        if (this->get_size().stores_equal_sizes()) {
            const auto entry_size =
                std::min(this->get_size().at(0)[0], this->get_size().at(0)[1]);
            return values_.get_data() + batch * entry_size;
        } else {
            return values_.get_data() +
                   num_elems_per_batch_cumul_.get_const_data()[batch];
        }
    }

    /**
     * @copydoc get_values()
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
     * @copydoc get_values(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_values(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        if (this->get_size().stores_equal_sizes()) {
            const auto entry_size =
                std::min(this->get_size().at(0)[0], this->get_size().at(0)[1]);
            return values_.get_const_data() + batch * entry_size;
        } else {
            return values_.get_const_data() +
                   num_elems_per_batch_cumul_.get_const_data()[batch];
        }
    }

    /**
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batches.
     *
     * @return the number of elements explicitly stored in the matrix,
     *         cumulative across all the batches
     */
    size_type get_num_stored_elements() const noexcept
    {
        return values_.get_num_elems();
    }

    /**
     * Returns the number of elements explicitly stored at a specific batch
     * index.
     *
     * @param batch  the batch index to be queried
     *
     * @return the number of elements explicitly stored in the matrix
     */
    size_type get_num_stored_elements(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        if (this->get_size().stores_equal_sizes()) {
            return std::min(this->get_size().at(0)[0],
                            this->get_size().at(0)[1]);
        } else {
            return num_elems_per_batch_cumul_.get_const_data()[batch + 1] -
                   num_elems_per_batch_cumul_.get_const_data()[batch];
        }
    }

    /**
     * Returns a single element for a particular batch entry.
     *
     * @param batch  the batch index to be queried
     * @param idx  a linear index of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the host results in a runtime error)
     * @note  This method may not be the fastest way to access matrix entries.
     *        @sa get_values, get_const_values
     */
    ValueType& at(size_type batch, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch, idx)];
    }

    /**
     * @copydoc BatchDiagonal::at(size_type, size_type)
     */
    ValueType at(size_type batch, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch, idx)];
    }

private:
    /**
     * Compute the memory required for the values array from the sizes and the
     * strides.
     */
    size_type compute_batch_mem(const batch_dim<2>& sizes) const
    {
        if (sizes.stores_equal_sizes()) {
            const size_type n_values_per_entry =
                std::min(sizes.at(0)[0], sizes.at(0)[1]);
            return n_values_per_entry * sizes.get_num_batch_entries();
        }
        size_type mem_req = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            mem_req += std::min(sizes.at(i)[0], sizes.at(i)[1]);
        }
        return mem_req;
    }

    /**
     * Extract sizes from the vector of the distinct Diagonal matrices.
     */
    batch_dim<2> get_sizes_from_mtxs(
        const std::vector<Diagonal<ValueType>*> mtxs) const
    {
        auto sizes = std::vector<dim<2>>(mtxs.size());
        for (auto i = 0; i < mtxs.size(); ++i) {
            sizes[i] = mtxs[i]->get_size();
        }
        return batch_dim<2>(sizes);
    }

    /**
     * Compute the number of elements stored in each batch and store it in a
     * prefixed sum fashion
     */
    inline array<size_type> compute_num_elems_per_batch_cumul(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes)
    {
        auto num_elems = array<size_type>(exec->get_master(),
                                          sizes.get_num_batch_entries() + 1);
        num_elems.get_data()[0] = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            num_elems.get_data()[i + 1] =
                num_elems.get_data()[i] +
                std::min(sizes.at(i)[0], sizes.at(i)[1]);
        }
        num_elems.set_executor(exec);
        return num_elems;
    }

protected:
    /**
     * Creates an uninitialized BatchDiagonal matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     */
    BatchDiagonal(std::shared_ptr<const Executor> exec,
                  const batch_dim<2>& size = batch_dim<2>{})
        : EnableBatchLinOp<BatchDiagonal>(exec, size),
          values_(exec, compute_batch_mem(size))
    {
        if (!size.stores_equal_sizes()) {
            num_elems_per_batch_cumul_ =
                compute_num_elems_per_batch_cumul(exec, this->get_size());
        }
    }

    /**
     * Creates a BatchDiagonal matrix from an already allocated (and
     * initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  sizes of the batch matrices in a batch_dim object
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
    BatchDiagonal(std::shared_ptr<const Executor> exec,
                  const batch_dim<2>& size, ValuesArray&& values)
        : EnableBatchLinOp<BatchDiagonal>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)}
    {
        if (!size.stores_equal_sizes()) {
            num_elems_per_batch_cumul_ =
                compute_num_elems_per_batch_cumul(exec, this->get_size());
        }
    }

    /**
     * Creates a BatchDiagonal matrix from a vector of matrices
     *
     * @param exec  Executor associated to the matrix
     * @param matrices  The matrices that need to be batched.
     */
    BatchDiagonal(std::shared_ptr<const Executor> exec,
                  const std::vector<Diagonal<ValueType>*>& matrices)
        : EnableBatchLinOp<BatchDiagonal>(exec, get_sizes_from_mtxs(matrices)),
          values_(exec, compute_batch_mem(this->get_size()))
    {
        num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
            exec->get_master(), this->get_size());
        for (size_type i = 0; i < this->get_num_batch_entries(); ++i) {
            auto local_exec = matrices[i]->get_executor();
            exec->copy_from(local_exec.get(), matrices[i]->get_size()[0],
                            matrices[i]->get_const_values(),
                            this->get_values() +
                                num_elems_per_batch_cumul_.get_const_data()[i]);
        }
    }

    /**
     * Creates a BatchDiagonal matrix by duplicating BatchDiagonal matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchDiagonal(std::shared_ptr<const Executor> exec,
                  const size_type num_duplications,
                  const BatchDiagonal<value_type>* const input)
        : EnableBatchLinOp<BatchDiagonal>(
              exec, gko::batch_dim<2>(
                        input->get_num_batch_entries() * num_duplications,
                        input->get_size().at(0)))
    {
        const auto in_batch_entries = input->get_num_batch_entries();
        const bool non_uniform = !input->get_size().stores_equal_sizes();
        std::vector<dim<2>> nu_dim;
        if (non_uniform) {
            nu_dim.resize(input->get_num_batch_entries() * num_duplications);
            for (size_type i = 0; i < num_duplications; ++i) {
                for (size_type batch_idx = i * in_batch_entries;
                     batch_idx < (i + 1) * in_batch_entries; batch_idx++) {
                    nu_dim[batch_idx] =
                        input->get_size().at(batch_idx - i * in_batch_entries);
                }
            }
            this->set_size(batch_dim<2>(nu_dim));
            num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
                exec->get_master(), this->get_size());
        }
        values_ = array<ValueType>(exec, compute_batch_mem(this->get_size()));
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            if (non_uniform) {
                for (size_type batch_idx = i * in_batch_entries;
                     batch_idx < (i + 1) * in_batch_entries; batch_idx++) {
                    nu_dim[batch_idx] =
                        input->get_size().at(batch_idx - i * in_batch_entries);
                }
            }
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_values(), this->get_values() + offset);
            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchDiagonal matrix by duplicating Diagonal matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchDiagonal(std::shared_ptr<const Executor> exec,
                  const size_type num_duplications,
                  const Diagonal<value_type>* const input)
        : EnableBatchLinOp<BatchDiagonal>(
              exec, gko::batch_dim<2>(num_duplications, input->get_size())),
          values_(exec, compute_batch_mem(this->get_size()))
    {
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(input->get_executor().get(), input->get_size()[0],
                            input->get_const_values(),
                            this->get_values() + offset);
            offset += input->get_size()[0];
        }
    }

    /**
     * Creates a BatchDiagonal matrix with the same configuration as the callers
     * matrix.
     *
     * @returns a BatchDiagonal matrix with the same configuration as the
     * caller.
     */
    virtual std::unique_ptr<BatchDiagonal> create_with_same_config() const
    {
        return BatchDiagonal::create(this->get_executor(), this->get_size());
    }

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

    size_type linearize_index(const size_type batch,
                              const size_type row) const noexcept
    {
        if (this->get_size().stores_equal_sizes()) {
            return row + batch * std::min(this->get_size().at(0)[0],
                                          this->get_size().at(0)[1]);
        } else {
            return num_elems_per_batch_cumul_.get_const_data()[batch] + row;
        }
    }

private:
    array<value_type> values_;
    array<size_type> num_elems_per_batch_cumul_;
};


/**
 * Transforms the input matrix A according to
 * S_L*A*S_R where '*' denotes matrix multiplication, and S_L and S_R
 * are the left and right transormation matrices.
 *
 * @param exec  Exector to run the operation on.
 * @param left  Left transformation matrix.
 * @param right  Right transformation matrix.
 * @param mtx  System matrix to be transformed.
 */
template <typename ValueType>
void two_sided_batch_transform(std::shared_ptr<const Executor> exec,
                               const BatchDiagonal<ValueType>* left,
                               const BatchDiagonal<ValueType>* right,
                               BatchLinOp* mtx);


/**
 * Transforms the input matrix A and vector b according to
 * S_L*A*S_R and S_L*b where '*' denotes matrix multiplication, and S_L and S_R
 * are the left and right transormation matrices.
 *
 * @param exec  Exector to run the operation on.
 * @param left  Left transformation matrix.
 * @param right  Right transformation matrix.
 * @param mtx  System matrix to be transformed.
 * @param rhs  Right-hand side of the sytem to be transformed.
 */
template <typename ValueType>
void two_sided_batch_system_transform(std::shared_ptr<const Executor> exec,
                                      const BatchDiagonal<ValueType>* left,
                                      const BatchDiagonal<ValueType>* right,
                                      BatchLinOp* mtx,
                                      BatchDense<ValueType>* rhs);


}  // namespace matrix


/**
 * Creates and initializes a batch of diagonal matrices.
 *
 * This function first creates a temporary matrix, fills it with passed in
 * values, and then copies the matrix to the requested backend.
 *
 * @tparam Matrix  matrix type to initialize
 *
 * @param vals  values used to initialize the batch vector
 * @param exec  Executor associated to the vector
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename ValueType>
std::unique_ptr<matrix::BatchDiagonal<ValueType>> batch_diagonal_initialize(
    std::initializer_list<std::initializer_list<ValueType>> vals,
    std::shared_ptr<const Executor> exec)
{
    using batch_diag = matrix::BatchDiagonal<ValueType>;
    size_type num_batch_entries = vals.size();
    std::vector<size_type> num_rows(num_batch_entries);
    std::vector<dim<2>> sizes(num_batch_entries);
    auto vals_begin = begin(vals);
    for (size_type b = 0; b < num_batch_entries; ++b) {
        num_rows[b] = vals_begin->size();
        sizes[b] = dim<2>(num_rows[b], num_rows[b]);
        vals_begin++;
    }
    auto b_size = batch_dim<2>(sizes);
    auto tmp = batch_diag::create(exec->get_master(), b_size);
    size_type batch = 0;
    for (const auto& b : vals) {
        size_type idx = 0;
        for (const auto& elem : b) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
        ++batch;
    }
    auto mtx = batch_diag::create(exec);
    tmp->move_to(mtx.get());
    return mtx;
}


/**
 * Creates and initializes a batch diagonal matrix by making copies of the
 * single input diagonal matrix.
 *
 * This function first creates a temporary batch matrix, fills it with
 * passed in values, and then copies the matrix to the requested backend.
 *
 * @tparam Matrix  matrix type to initialize
 *
 * @param num_matrices  The number of times the input vector is copied into
 *                     the final output
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated to the vector
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename ValueType>
std::unique_ptr<matrix::BatchDiagonal<ValueType>> batch_diagonal_initialize(
    const size_type num_matrices, std::initializer_list<ValueType> vals,
    std::shared_ptr<const Executor> exec)
{
    using batch_diag = matrix::BatchDiagonal<ValueType>;
    std::vector<size_type> num_rows(num_matrices);
    std::vector<dim<2>> sizes(num_matrices);
    for (size_type b = 0; b < num_matrices; ++b) {
        num_rows[b] = vals.size();
        sizes[b] = dim<2>(vals.size(), vals.size());
    }
    auto b_size = batch_dim<2>(sizes);
    auto tmp = batch_diag::create(exec->get_master(), b_size);
    for (size_type batch = 0; batch < num_matrices; batch++) {
        size_type idx = 0;
        for (const auto& elem : vals) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
    }
    auto mtx = batch_diag::create(exec);
    tmp->move_to(mtx.get());
    return mtx;
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DIAGONAL_HPP_
