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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_BAND_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_BAND_HPP_


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


template <typename ValueType, typename IndexType>
class BatchCsr;

template <typename ValueType>
class BatchDense;


/**
 * BatchBand is a matrix format that holds a collection of banded
 * matrices.
 *
 *
 * @tparam ValueType  precision of matrix elements
 *
 *
 * @ingroup batch_band
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchBand : public EnableBatchLinOp<BatchBand<ValueType>>,
                  public EnableCreateMethod<BatchBand<ValueType>>,
                  public ConvertibleTo<BatchBand<next_precision<ValueType>>>,
                  public ConvertibleTo<BatchCsr<ValueType, int32>>,
                  public ConvertibleTo<BatchDense<ValueType>>,
                  public BatchReadableFromMatrixData<ValueType, int32>,
                  public BatchReadableFromMatrixData<ValueType, int64>,
                  public BatchWritableToMatrixData<ValueType, int32>,
                  public BatchWritableToMatrixData<ValueType, int64>,
                  public BatchTransposable,
                  public BatchScaledIdentityAddable {
    friend class EnableCreateMethod<BatchBand>;
    friend class EnablePolymorphicObject<BatchBand, BatchLinOp>;
    friend class BatchBand<to_complex<ValueType>>;

public:
    using EnableBatchLinOp<BatchBand>::convert_to;
    using EnableBatchLinOp<BatchBand>::move_to;
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = BatchBand<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchBand>;
    using complex_type = to_complex<BatchBand>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a BatchBand matrix with the configuration of another
     * BatchBand matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchBand> create_with_config_of(
        const BatchBand* other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    friend class BatchBand<next_precision<ValueType>>;

    void convert_to(
        BatchBand<next_precision<ValueType>>* result) const override;

    void move_to(BatchBand<next_precision<ValueType>>* result) override;

    void convert_to(BatchCsr<ValueType, index_type>* result) const override;

    void move_to(BatchCsr<ValueType, index_type>* result) override;

    void convert_to(BatchDense<ValueType>* result) const override;

    void move_to(BatchDense<ValueType>* result) override;

    void read(const std::vector<mat_data>& data) override;

    void read(const std::vector<mat_data32>& data) override;

    void write(std::vector<mat_data>& data) const override;

    void write(std::vector<mat_data32>& data) const override;

    std::unique_ptr<BatchLinOp> transpose() const override;

    std::unique_ptr<BatchLinOp> conj_transpose() const override;

    bool check_if_element_is_part_of_the_band(
        size_type batch, size_type dense_row,
        size_type dense_col) const noexcept
    {
        const auto n = this->get_size().at(batch)[0];
        const auto kl = KL_.at(batch);
        const auto ku = KU_.at(batch);

        if (dense_row >= std::max(size_type{0}, dense_col - ku) ||
            dense_row <= std::min(n - 1, dense_col + kl)) {
            return true;
        } else {
            return false;
        }
    }

    size_type get_linear_index_wrt_band_arr(size_type batch,
                                            size_type dense_row,
                                            size_type dense_col) const
    {
        const auto n = this->get_size().at(batch)[0];
        const auto kl = KL_.at(batch);
        const auto ku = KU_.at(batch);

        if (!check_if_element_is_part_of_the_band(batch, dense_row,
                                                  dense_col)) {
            throw std::runtime_error(
                "Requested element is not a part of the band!");
        }

        const auto band_row = kl + ku + dense_row - dense_col;
        const auto band_col = dense_col;
        return num_elems_per_batch_cumul_.get_const_data()[batch] + band_row +
               band_col * (2 * kl + ku + 1);
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
    value_type& at(size_type batch, size_type row_in_dense_layout,
                   size_type col_in_dense_layout)
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return band_array_col_major_.get_data()[get_linear_index_wrt_band_arr(
            batch, row_in_dense_layout, col_in_dense_layout)];
    }

    /**
     * @copydoc BatchBand::at(size_type, size_type, size_type)
     */
    value_type at(size_type batch, size_type row_in_dense_layout,
                  size_type col_in_dense_layout) const
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return band_array_col_major_
            .get_const_data()[get_linear_index_wrt_band_arr(
                batch, row_in_dense_layout, col_in_dense_layout)];
    }

    /**
     * Returns a pointer to the array of bands of the batched matrix.
     *
     * @return the pointer to the band array
     */
    value_type* get_band_array() noexcept
    {
        return band_array_col_major_.get_data();
    }

    /**
     * Returns a pointer to the array of bands of the batched matrix.
     *
     * @return the pointer to the band array
     */
    value_type* get_band_array(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return band_array_col_major_.get_data() +
               num_elems_per_batch_cumul_.get_const_data()[batch];
    }

    /**
     * @copydoc get_band_array()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_band_array() const noexcept
    {
        return band_array_col_major_.get_const_data();
    }

    /**
     * @copydoc get_band_array(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_band_array(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return band_array_col_major_.get_const_data() +
               num_elems_per_batch_cumul_.get_const_data()[batch];
    }

    batch_stride get_num_lower_diagonals() const noexcept { return KL_; }

    batch_stride get_num_upper_diagonals() const noexcept { return KU_; }

    /**
     * Returns the number of elements explicitly stored in the batch matrix,
     * cumulative across all the batches.
     *
     * @return the number of elements explicitly stored in the matrix,
     *         cumulative across all the batches
     */
    size_type get_num_stored_elements() const noexcept
    {
        return band_array_col_major_.get_num_elems();
    }

    /**
     * Returns the number of elements stored for a particular batch
     * entry
     *
     * @return the number of elements stored for a particular batch
     * entry
     *
     */
    size_type get_num_stored_elements(const size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return num_elems_per_batch_cumul_.get_const_data()[batch + 1] -
               num_elems_per_batch_cumul_.get_const_data()[batch];
    }

    /**
     * Creates a constant (immutable) batch tridiagonal matrix from a constant
     * array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param KLs
     * @param Kus
     * @param band_arr_col_major
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const BatchBand> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        const batch_stride& KLs, const batch_stride& KUs,
        gko::detail::const_array_view<ValueType>&& band_arr_col_major)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const BatchBand>(new BatchBand{
            exec, sizes, KLs, KUs,
            gko::detail::array_const_cast(std::move(band_arr_col_major))});
    }

private:
    /**
     * Compute the total memory required to store the batched banded matrix from
     * the sizes, KLs and KUs.
     */
    inline size_type compute_batch_mem(const batch_dim<2>& sizes,
                                       const batch_stride& KLs,
                                       const batch_stride& KUs)
    {
        if (sizes.stores_equal_sizes() && KLs.stores_equal_strides() &&
            KUs.stores_equal_strides()) {
            return sizes.at(0)[0] * (2 * KLs.at(0) + KUs.at(0) + 1) *
                   sizes.get_num_batch_entries();
        }
        size_type mem_req = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            mem_req += sizes.at(i)[0] * (2 * KLs.at(i) + KUs.at(i) + 1);
        }
        return mem_req;
    }

    /**
     * Compute the number of elements stored for each batch entry
     * and store it in a prefixed sum fashion
     */
    inline array<size_type> compute_num_elems_per_batch_cumul(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        const batch_stride& KLs, const batch_stride& KUs)
    {
        auto num_elems = array<size_type>(exec->get_master(),
                                          sizes.get_num_batch_entries() + 1);
        num_elems.get_data()[0] = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            num_elems.get_data()[i + 1] =
                num_elems.get_data()[i] +
                sizes.at(i)[0] * (2 * KLs.at(i) + KUs.at(i) + 1);
        }
        num_elems.set_executor(exec);
        return num_elems;
    }

protected:
    /**
     * Creates an uninitialized BatchBand matrix of the specified size, KL and
     * KU.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     * @param KL
     * @param KU
     *
     */
    BatchBand(std::shared_ptr<const Executor> exec,
              const batch_dim<2>& size = batch_dim<2>{},
              const batch_stride& KL = batch_stride{},
              const batch_stride& KU = batch_stride{})
        : EnableBatchLinOp<BatchBand>(exec, size),
          band_array_col_major_(exec, compute_batch_mem(size, KL, KU)),
          KL_(KL),
          KU_(KU)
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        num_elems_per_batch_cumul_ =
            compute_num_elems_per_batch_cumul(exec, this->get_size(), KL, KU);
    }

    /**
     * Creates a BatchBand matrix from an already allocated (and
     * initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param KL
     * @param KU
     * @param band_arr_col_major band stored in dense column major array
     *
     * @note If 'band_arr_col_major' is not
     * an rvalue, not an array of ValueType or is on the wrong executor, an
     * internal copy of that array will be created, and the original array data
     * will not be used in the matrix.
     *
     */
    template <typename ValuesArray>
    BatchBand(std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
              const batch_stride& KL, const batch_stride& KU,
              ValuesArray&& band_arr_col_major)
        : EnableBatchLinOp<BatchBand>(exec, size),
          band_array_col_major_{exec,
                                std::forward<ValuesArray>(band_arr_col_major)},
          KL_(KL),
          KU_(KU),
          num_elems_per_batch_cumul_(
              exec->get_master(),
              compute_num_elems_per_batch_cumul(exec->get_master(),
                                                this->get_size(), KL, KU))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        auto num_elems =
            num_elems_per_batch_cumul_
                .get_const_data()[num_elems_per_batch_cumul_.get_num_elems() -
                                  1] -
            1;
        GKO_ENSURE_IN_BOUNDS(num_elems, band_array_col_major_.get_num_elems());
    }


    /**
     * Creates a BatchBand matrix by duplicating BatchBand matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    // NOTE: Currently, this works only for a uniform batch
    BatchBand(std::shared_ptr<const Executor> exec, size_type num_duplications,
              const BatchBand<value_type>* input)
        : EnableBatchLinOp<BatchBand>(
              exec, gko::batch_dim<2>(
                        input->get_num_batch_entries() * num_duplications,
                        input->get_size().at(0))),
          KL_{gko::batch_stride(
              input->get_num_batch_entries() * num_duplications,
              input->get_num_lower_diagonals().at(0))},
          KU_{gko::batch_stride(
              input->get_num_batch_entries() * num_duplications,
              input->get_num_upper_diagonals().at(0))},
          band_array_col_major_(exec,
                                compute_batch_mem(this->get_size(), KL_, KU_))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        num_elems_per_batch_cumul_ = compute_num_elems_per_batch_cumul(
            exec->get_master(), this->get_size(),
            this->get_num_lower_diagonals(), this->get_num_upper_diagonals());
        size_type offset = 0;

        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(
                input->get_executor().get(), input->get_num_stored_elements(),
                input->get_const_band_array(), this->get_band_array() + offset);

            offset += input->get_num_stored_elements();
        }
    }

    /**
     * Creates a BatchBand matrix with the same configuration as the
     * callers matrix.
     *
     * @returns a BatchBand matrix with the same configuration as the
     * caller.
     */
    virtual std::unique_ptr<BatchBand> create_with_same_config() const
    {
        return BatchBand::create(this->get_executor(), this->get_size());
    }

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

private:
    array<size_type> num_elems_per_batch_cumul_;
    array<value_type> band_array_col_major_;
    batch_stride KL_;
    batch_stride KU_;

    void add_scaled_identity_impl(const BatchLinOp* a,
                                  const BatchLinOp* b) override;
};


}  // namespace matrix


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_BAND_HPP_
