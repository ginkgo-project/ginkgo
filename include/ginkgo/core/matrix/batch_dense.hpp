/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_


#include <initializer_list>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_lin_op.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
class BatchDiagonal;


template <typename ValueType, typename IndexType>
class BatchCsr;


/**
 * BatchDense is a batch matrix format which explicitly stores all values of the
 * matrix in each of the batches.
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
class BatchDense : public EnableBatchLinOp<BatchDense<ValueType>>,
                   public EnableCreateMethod<BatchDense<ValueType>>,
                   public ConvertibleTo<BatchDense<next_precision<ValueType>>>,
                   public ConvertibleTo<BatchCsr<ValueType, int32>>,
                   public ConvertibleTo<BatchDiagonal<ValueType>>,
                   public BatchReadableFromMatrixData<ValueType, int32>,
                   public BatchReadableFromMatrixData<ValueType, int64>,
                   public BatchWritableToMatrixData<ValueType, int32>,
                   public BatchWritableToMatrixData<ValueType, int64>,
                   public BatchTransposable,
                   public BatchScaledIdentityAddable {
    friend class EnableCreateMethod<BatchDense>;
    friend class EnablePolymorphicObject<BatchDense, BatchLinOp>;
    friend class BatchDense<to_complex<ValueType>>;

public:
    using EnableBatchLinOp<BatchDense>::convert_to;
    using EnableBatchLinOp<BatchDense>::move_to;
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = BatchDense<ValueType>;
    using unbatch_type = Dense<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchDense>;
    using complex_type = to_complex<BatchDense>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a BatchDense matrix with the configuration of another BatchDense
     * matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchDense> create_with_config_of(
        const BatchDense* other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    friend class BatchDense<next_precision<ValueType>>;

    void convert_to(
        BatchDense<next_precision<ValueType>>* result) const override;

    void move_to(BatchDense<next_precision<ValueType>>* result) override;

    void convert_to(BatchCsr<ValueType, index_type>* result) const override;

    void move_to(BatchCsr<ValueType, index_type>* result) override;

    void convert_to(BatchDiagonal<ValueType>* result) const override;

    void move_to(BatchDiagonal<ValueType>* result) override;

    void read(const std::vector<mat_data>& data) override;

    void read(const std::vector<mat_data32>& data) override;

    void write(std::vector<mat_data>& data) const override;

    void write(std::vector<mat_data32>& data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    /**
     * Unbatches the batched dense and creates a std::vector of Dense matrices
     *
     * @return  a std::vector containing the Dense matrices.
     */
    std::vector<std::unique_ptr<unbatch_type>> unbatch() const
    {
        auto exec = this->get_executor();
        auto unbatch_mats = std::vector<std::unique_ptr<unbatch_type>>{};
        for (size_type b = 0; b < this->get_num_batch_entries(); ++b) {
            auto mat = unbatch_type::create(exec, this->get_size().at(b),
                                            this->get_stride().at(b));
            exec->copy_from(exec.get(), mat->get_num_stored_elements(),
                            this->get_const_values() +
                                num_elems_per_batch_cumul_.get_const_data()[b],
                            mat->get_values());
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
        return values_.get_data() +
               num_elems_per_batch_cumul_.get_const_data()[batch];
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
        return values_.get_const_data() +
               num_elems_per_batch_cumul_.get_const_data()[batch];
    }

    /**
     * Returns the batch_stride of the matrix.
     *
     * @return the batch_stride of the matrix.
     */
    const batch_stride& get_stride() const noexcept { return stride_; }

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
        return num_elems_per_batch_cumul_.get_const_data()[batch + 1] -
               num_elems_per_batch_cumul_.get_const_data()[batch];
    }

    /**
     * Returns a single element for a particular batch.
     *
     * @param batch  the batch index to be queried
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& at(size_type batch, size_type row, size_type col) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_data()[linearize_index(batch, row, col)];
    }

    /**
     * @copydoc BatchDense::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch, size_type row, size_type col) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return values_.get_const_data()[linearize_index(batch, row, col)];
    }

    /**
     * Returns a single element for a particular batch entry.
     *
     * Useful for iterating across all elements of the matrix.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param batch  the batch index to be queried
     * @param idx  a linear index of the requested element
     *             (ignoring the stride)
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    ValueType& at(size_type batch, size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(batch, idx)];
    }

    /**
     * @copydoc BatchDense::at(size_type, size_type, size_type)
     */
    ValueType at(size_type batch, size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(batch, idx)];
    }

    /**
     * Scales the matrix with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 BatchDense matrix, the entire matrix (all
     * batches) is scaled by alpha. If it is a BatchDense row vector of values,
     * then i-th column of the matrix is scaled with the i-th element of alpha
     * (the number of columns of alpha has to match the number of columns of the
     * matrix).
     */
    void scale(const BatchLinOp* alpha)
    {
        auto exec = this->get_executor();
        this->scale_impl(make_temporary_clone(exec, alpha).get());
    }

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 BatchDense matrix, the entire matrix is
     * scaled by alpha. If it is a BatchDense row vector of values, then i-th
     * column of the matrix is scaled with the i-th element of alpha (the number
     * of columns of alpha has to match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    void add_scaled(const BatchLinOp* alpha, const BatchLinOp* b)
    {
        auto exec = this->get_executor();
        this->add_scaled_impl(make_temporary_clone(exec, alpha).get(),
                              make_temporary_clone(exec, b).get());
    }

    /**
     * Adds `a` scaled by `alpha` to the matrix scaled by `beta`:
     * this <- alpha * a + beta * this.
     *
     * @param alpha  If alpha is 1x1 BatchDense matrix, the entire matrix a is
     *               scaled by alpha. If it is a BatchDense row vector of
     *               values, then i-th column of a is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of a).
     * @param a  a matrix of the same dimension as this.
     * @param beta  Scalar(s), of the same size as alpha, to multiply this
     * matrix.
     */
    void add_scale(const BatchLinOp* alpha, const BatchLinOp* a,
                   const BatchLinOp* beta);

    /**
     * Computes the column-wise dot product of each matrix in this batch and its
     * corresponding entry in `b`. If the matrix has complex value_type, then
     * the conjugate of this is taken.
     *
     * @param b  a BatchDense matrix of same dimension as this
     * @param result  a BatchDense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_dot(const BatchLinOp* b, BatchLinOp* result) const
    {
        auto exec = this->get_executor();
        this->compute_dot_impl(make_temporary_clone(exec, b).get(),
                               make_temporary_clone(exec, result).get());
    }

    /**
     * Computes the Euclidean (L^2) norm of each matrix in this batch.
     *
     * @param result  a BatchDense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(BatchLinOp* result) const
    {
        auto exec = this->get_executor();
        this->compute_norm2_impl(make_temporary_clone(exec, result).get());
    }

    /**
     * Creates a constant (immutable) batch dense matrix from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param stride  the row-stride of the matrix
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const BatchDense> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& values,
        const batch_stride& strides)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const BatchDense>(new BatchDense{
            exec, sizes, gko::detail::array_const_cast(std::move(values)),
            strides});
    }

private:
    /**
     * Compute the memory required for the values array from the sizes and the
     * strides.
     */
    inline size_type compute_batch_mem(const batch_dim<2>& sizes,
                                       const batch_stride& strides)
    {
        GKO_ASSERT(sizes.get_num_batch_entries() ==
                   strides.get_num_batch_entries());
        if (sizes.stores_equal_sizes() && strides.stores_equal_strides()) {
            return (sizes.at(0))[0] * strides.at(0) *
                   sizes.get_num_batch_entries();
        }
        size_type mem_req = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            mem_req += (sizes.at(i))[0] * strides.at(i);
        }
        return mem_req;
    }

    /**
     * Extract the nth dim of the batch sizes from the input batch_dim object.
     */
    inline batch_stride extract_nth_dim(const int dim, const batch_dim<2>& size)
    {
        if (size.stores_equal_sizes()) {
            return batch_stride(size.get_num_batch_entries(), size.at(0)[dim]);
        }
        std::vector<size_type> stride(size.get_num_batch_entries());
        for (auto i = 0; i < size.get_num_batch_entries(); ++i) {
            stride[i] = (size.at(i))[dim];
        }
        return batch_stride(stride);
    }

    /**
     * Extract strides from the vector of the distinct Dense matrices.
     */
    inline batch_stride get_strides_from_mtxs(
        const std::vector<Dense<ValueType>*> mtxs)
    {
        auto strides = std::vector<size_type>(mtxs.size());
        for (auto i = 0; i < mtxs.size(); ++i) {
            strides[i] = mtxs[i]->get_stride();
        }
        return batch_stride(strides);
    }

    /**
     * Extract sizes from the vector of the distinct Dense matrices.
     */
    inline batch_dim<2> get_sizes_from_mtxs(
        const std::vector<Dense<ValueType>*> mtxs)
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
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        const batch_stride& strides)
    {
        auto num_elems = array<size_type>(exec->get_master(),
                                          sizes.get_num_batch_entries() + 1);
        num_elems.get_data()[0] = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            num_elems.get_data()[i + 1] =
                num_elems.get_data()[i] + (sizes.at(i))[0] * strides.at(i);
        }
        num_elems.set_executor(exec);
        return num_elems;
    }

protected:
    /**
     * Creates an uninitialized BatchDense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    BatchDense(std::shared_ptr<const Executor> exec,
               const batch_dim<2>& size = batch_dim<2>{})
        : BatchDense(std::move(exec), size,
                     size.get_num_batch_entries() > 0 ? extract_nth_dim(1, size)
                                                      : batch_stride{})
    {}

    /**
     * Creates an uninitialized BatchDense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    BatchDense(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
               const batch_stride& stride)
        : EnableBatchLinOp<BatchDense>(exec, size),
          values_(exec, compute_batch_mem(size, stride)),
          stride_(stride)
    {
        num_elems_per_batch_cumul_ =
            compute_num_elems_per_batch_cumul(exec, this->get_size(), stride);
    }

    /**
     * Creates a BatchDense matrix from an already allocated (and initialized)
     * array.
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
    BatchDense(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
               ValuesArray&& values, const batch_stride& stride)
        : EnableBatchLinOp<BatchDense>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          stride_{stride},
          num_elems_per_batch_cumul_(
              exec->get_master(),
              compute_num_elems_per_batch_cumul(exec->get_master(),
                                                this->get_size(), stride))
    {
        auto num_elems =
            num_elems_per_batch_cumul_
                .get_const_data()[num_elems_per_batch_cumul_.get_num_elems() -
                                  1] -
            1;
        GKO_ENSURE_IN_BOUNDS(num_elems, values_.get_num_elems());
    }

    /**
     * Creates a BatchDense matrix from a vector of matrices
     *
     * @param exec  Executor associated to the matrix
     * @param matrices  The matrices that need to be batched.
     */
    BatchDense(std::shared_ptr<const Executor> exec,
               const std::vector<Dense<ValueType>*>& matrices)
        : EnableBatchLinOp<BatchDense>(exec, get_sizes_from_mtxs(matrices)),
          stride_{get_strides_from_mtxs(matrices)},
          values_(exec, compute_batch_mem(this->get_size(), stride_))
    {
        num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
            exec->get_master(), this->get_size(), stride_);
        for (size_type i = 0; i < this->get_num_batch_entries(); ++i) {
            auto local_exec = matrices[i]->get_executor();
            exec->copy_from(local_exec.get(),
                            matrices[i]->get_num_stored_elements(),
                            matrices[i]->get_const_values(),
                            this->get_values() +
                                num_elems_per_batch_cumul_.get_const_data()[i]);
        }
    }

    /**
     * Creates a BatchDense matrix by duplicating BatchDense matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchDense(std::shared_ptr<const Executor> exec, size_type num_duplications,
               const BatchDense<value_type>* input)
        : EnableBatchLinOp<BatchDense>(
              exec, gko::batch_dim<2>(
                        input->get_num_batch_entries() * num_duplications,
                        input->get_size().at(0))),
          stride_{gko::batch_stride(
              input->get_num_batch_entries() * num_duplications,
              input->get_stride().at(0))},
          values_(exec, compute_batch_mem(this->get_size(), stride_))
    {
        // Check if it works when stride neq num_cols
        num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
            exec->get_master(), this->get_size(), stride_);
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_values(), this->get_values() + offset);
            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchDense matrix by duplicating Dense matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    BatchDense(std::shared_ptr<const Executor> exec, size_type num_duplications,
               const Dense<value_type>* input)
        : EnableBatchLinOp<BatchDense>(
              exec, gko::batch_dim<2>(num_duplications, input->get_size())),
          stride_{gko::batch_stride(num_duplications, input->get_stride())},
          values_(exec, compute_batch_mem(this->get_size(), stride_))
    {
        // Check if it works when stride neq num_cols
        num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
            exec->get_master(), this->get_size(), stride_);
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_values(), this->get_values() + offset);
            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchDense matrix with the same configuration as the callers
     * matrix.
     *
     * @returns a BatchDense matrix with the same configuration as the caller.
     */
    virtual std::unique_ptr<BatchDense> create_with_same_config() const
    {
        return BatchDense::create(this->get_executor(), this->get_size(),
                                  this->get_stride());
    }

    /**
     * @copydoc scale(const BatchLinOp *)
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of scale(const BatchLinOp *alpha).
     */
    virtual void scale_impl(const BatchLinOp* alpha);

    /**
     * @copydoc add_scaled(const BatchLinOp *, const BatchLinOp *)
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of add_scale(const BatchLinOp *alpha, const BatchLinOp
     * *b).
     */
    virtual void add_scaled_impl(const BatchLinOp* alpha, const BatchLinOp* b);

    /**
     * @copydoc compute_dot(const BatchLinOp *, BatchLinOp *) const
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of compute_dot(const BatchLinOp *b, BatchLinOp *result).
     */
    virtual void compute_dot_impl(const BatchLinOp* b,
                                  BatchLinOp* result) const;

    /**
     * @copydoc compute_norm2(BatchLinOp *) const
     *
     * @note  Other implementations of batch_dense should override this function
     *        instead of compute_norm2(BatchLinOp *result).
     */
    virtual void compute_norm2_impl(BatchLinOp* result) const;

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

    size_type linearize_index(size_type batch, size_type row,
                              size_type col) const noexcept
    {
        return num_elems_per_batch_cumul_.get_const_data()[batch] +
               row * stride_.at(batch) + col;
    }

    size_type linearize_index(size_type batch, size_type idx) const noexcept
    {
        return linearize_index(batch, idx / this->get_size().at(batch)[1],
                               idx % this->get_size().at(batch)[1]);
    }

private:
    batch_stride stride_;
    array<size_type> num_elems_per_batch_cumul_;
    array<value_type> values_;

    void add_scaled_identity_impl(const BatchLinOp* a,
                                  const BatchLinOp* b) override;
};


}  // namespace matrix


/**
 * Creates and initializes a batch of column-vectors.
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
 * @param vals  values used to initialize the batch vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::vector<size_type> stride,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    size_type num_batch_entries = vals.size();
    std::vector<size_type> num_rows(num_batch_entries);
    std::vector<dim<2>> sizes(num_batch_entries);
    auto vals_begin = begin(vals);
    for (size_type b = 0; b < num_batch_entries; ++b) {
        num_rows[b] = vals_begin->size();
        sizes[b] = dim<2>(num_rows[b], 1);
        vals_begin++;
    }
    auto b_size = batch_dim<2>(sizes);
    auto b_stride = batch_stride(stride);
    auto tmp = batch_dense::create(exec->get_master(), b_size, b_stride);
    size_type batch = 0;
    for (const auto& b : vals) {
        size_type idx = 0;
        for (const auto& elem : b) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
        ++batch;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}

/**
 * Creates and initializes a batch of column-vectors.
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
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return batch_initialize<Matrix>(std::vector<size_type>(vals.size(), 1),
                                    vals, std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a batch of matrices.
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
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::vector<size_type> stride,
    std::initializer_list<std::initializer_list<
        std::initializer_list<typename Matrix::value_type>>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    size_type num_batch_entries = vals.size();
    std::vector<size_type> num_rows(num_batch_entries);
    std::vector<size_type> num_cols(num_batch_entries);
    std::vector<dim<2>> sizes(num_batch_entries);
    size_type ind = 0;
    for (const auto& b : vals) {
        num_rows[ind] = b.size();
        num_cols[ind] = num_rows[ind] > 0 ? begin(b)->size() : 1;
        sizes[ind] = dim<2>(num_rows[ind], num_cols[ind]);
        ++ind;
    }
    auto b_size = batch_dim<2>(sizes);
    auto b_stride = batch_stride(stride);
    auto tmp = batch_dense::create(exec->get_master(), b_size, b_stride);
    size_type batch = 0;
    for (const auto& b : vals) {
        size_type ridx = 0;
        for (const auto& row : b) {
            size_type cidx = 0;
            for (const auto& elem : row) {
                tmp->at(batch, ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
        ++batch;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}


/**
 * Creates and initializes a batch of matrices.
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
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::initializer_list<std::initializer_list<
        std::initializer_list<typename Matrix::value_type>>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    auto strides = std::vector<size_type>(vals.size(), 0);
    size_type ind = 0;
    for (const auto& b : vals) {
        strides[ind] = begin(b)->size();
        ++ind;
    }
    return batch_initialize<Matrix>(strides, vals, std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a batch column-vector by making copies of the single
 * input column vector.
 *
 * This function first creates a temporary batch dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                  interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row strides for the temporary batch dense matrix
 * @param num_vectors  The number of times the input vector is copied into
 *                     the final output
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::vector<size_type> stride, const size_type num_vectors,
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    std::vector<size_type> num_rows(num_vectors);
    std::vector<dim<2>> sizes(num_vectors);
    for (size_type b = 0; b < num_vectors; ++b) {
        num_rows[b] = vals.size();
        sizes[b] = dim<2>(vals.size(), 1);
    }
    auto b_size = batch_dim<2>(sizes);
    auto b_stride = batch_stride(stride);
    auto tmp = batch_dense::create(exec->get_master(), b_size, b_stride);
    for (size_type batch = 0; batch < num_vectors; batch++) {
        size_type idx = 0;
        for (const auto& elem : vals) {
            tmp->at(batch, idx) = elem;
            ++idx;
        }
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}


/**
 * Creates and initializes a column-vector from copies of a given vector.
 *
 * This function first creates a temporary Dense matrix, fills it with passed
 * in values, and then converts the matrix to the requested type. The stride of
 * the intermediate Dense matrix is set to 1.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix>
 *                  interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param num_vectors  The number of times the input vector is copied into
 *                     the final output
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup BatchLinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    const size_type num_vectors,
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return batch_initialize<Matrix>(std::vector<size_type>(num_vectors, 1),
                                    num_vectors, vals, std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}

/**
 * Creates and initializes a matrix from copies of a given matrix.
 *
 * This function first creates a temporary batch dense matrix, fills it with
 * passed in values, and then converts the matrix to the requested type.
 *
 * @tparam Matrix  matrix type to initialize
 *                 (Dense has to implement the ConvertibleTo<Matrix> interface)
 * @tparam TArgs  argument types for Matrix::create method
 *                (not including the implied Executor as the first argument)
 *
 * @param stride  row strides for the temporary batch dense matrix
 * @param num_matrices  The number of times the input matrix is copied into
 *                     the final output
 * @param vals  values used to initialize each vector in the temp. batch
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    std::vector<size_type> stride, const size_type num_matrices,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using batch_dense = matrix::BatchDense<typename Matrix::value_type>;
    std::vector<dim<2>> sizes(num_matrices);
    const size_type num_rows = vals.size();
    for (size_type b = 0; b < num_matrices; ++b) {
        const size_type num_cols = begin(vals)->size();
        sizes[b] = dim<2>(num_rows, num_cols);
        for (auto blockit = begin(vals); blockit != end(vals); ++blockit) {
            GKO_ASSERT(blockit->size() == num_cols);
        }
    }
    auto tmp = batch_dense::create(exec->get_master(), sizes, stride);
    for (size_type batch = 0; batch < num_matrices; batch++) {
        size_type ridx = 0;
        for (const auto& row : vals) {
            size_type cidx = 0;
            for (const auto& elem : row) {
                tmp->at(batch, ridx, cidx) = elem;
                ++cidx;
            }
            ++ridx;
        }
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx.get());
    return mtx;
}

/**
 * Creates and initializes a matrix from copies of a given matrix.
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
 * @param num_vectors  The number of times the input vector is copied into
 *                     the final output
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 * @ingroup mat_formats
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> batch_initialize(
    const size_type num_matrices,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    auto strides = std::vector<size_type>(num_matrices, begin(vals)->size());
    return batch_initialize<Matrix>(strides, num_matrices, vals,
                                    std::move(exec),
                                    std::forward<TArgs>(create_args)...);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_DENSE_HPP_
