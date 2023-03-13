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

#ifndef GKO_PUBLIC_CORE_MATRIX_BATCH_TRIDIAGONAL_HPP_
#define GKO_PUBLIC_CORE_MATRIX_BATCH_TRIDIAGONAL_HPP_


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
 * BatchTridiagonal is a batch matrix format which stores the sub-diagonal (with
 * an extra zero in the beginning), the main- diagonal and the super-diagonal
 * (with an extra zero at the end) of each matrix in the batch.
 *
 *
 * The diagonals are stored in a strided fashion, i.e. first all elements of the
 * sub-diagonal of the first matrix, then all the elements of the sub-diagonal
 * of the second matrix and so on.
 *
 *
 * @tparam ValueType  precision of matrix elements
 *
 *
 * @ingroup batch_tridiagonal
 * @ingroup mat_formats
 * @ingroup BatchLinOp
 */
template <typename ValueType = default_precision>
class BatchTridiagonal
    : public EnableBatchLinOp<BatchTridiagonal<ValueType>>,
      public EnableCreateMethod<BatchTridiagonal<ValueType>>,
      public ConvertibleTo<BatchTridiagonal<next_precision<ValueType>>>,
      public ConvertibleTo<BatchCsr<ValueType, int32>>,
      public ConvertibleTo<BatchDense<ValueType>>,
      public BatchReadableFromMatrixData<ValueType, int32>,
      public BatchReadableFromMatrixData<ValueType, int64>,
      public BatchWritableToMatrixData<ValueType, int32>,
      public BatchWritableToMatrixData<ValueType, int64>,
      public BatchTransposable,
      public BatchScaledIdentityAddable {
    friend class EnableCreateMethod<BatchTridiagonal>;
    friend class EnablePolymorphicObject<BatchTridiagonal, BatchLinOp>;
    friend class BatchTridiagonal<to_complex<ValueType>>;

public:
    using EnableBatchLinOp<BatchTridiagonal>::convert_to;
    using EnableBatchLinOp<BatchTridiagonal>::move_to;
    using BatchReadableFromMatrixData<ValueType, int32>::read;
    using BatchReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int32;
    using transposed_type = BatchTridiagonal<ValueType>;
    using mat_data = gko::matrix_data<ValueType, int64>;
    using mat_data32 = gko::matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<BatchTridiagonal>;
    using complex_type = to_complex<BatchTridiagonal>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a BatchTridiagonal matrix with the configuration of another
     * BatchTridiagonal matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<BatchTridiagonal> create_with_config_of(
        const BatchTridiagonal* other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    friend class BatchTridiagonal<next_precision<ValueType>>;

    void convert_to(
        BatchTridiagonal<next_precision<ValueType>>* result) const override;

    void move_to(BatchTridiagonal<next_precision<ValueType>>* result) override;

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

    /**
     * Returns a pointer to the array of sub-diagonals of the batched matrix.
     *
     * @return the pointer to the array of sub-diagonals
     */
    value_type* get_sub_diagonal() noexcept { return sub_diagonal_.get_data(); }

    /**
     * Returns a pointer to the array of sub-diagonals of the batched matrix.
     *
     * @return the pointer to the array of sub-diagonals
     */
    value_type* get_sub_diagonal(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return sub_diagonal_.get_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }


    /**
     * Returns a pointer to the array of main diagonals of the batched matrix.
     *
     * @return the pointer to the array of main diagonals
     */
    value_type* get_main_diagonal() noexcept
    {
        return main_diagonal_.get_data();
    }

    /**
     * Returns a pointer to the array of main diagonals of the batched matrix.
     *
     * @return the pointer to the array of main diagonals
     */
    value_type* get_main_diagonal(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return main_diagonal_.get_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }

    /**
     * Returns a pointer to the array of super-diagonals of the batched matrix.
     *
     * @return the pointer to the array of super-diagonals
     */
    value_type* get_super_diagonal() noexcept
    {
        return super_diagonal_.get_data();
    }

    /**
     * Returns a pointer to the array of super-diagonals of the batched matrix.
     *
     * @return the pointer to the array of super-diagonal
     */
    value_type* get_super_diagonal(size_type batch) noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return super_diagonal_.get_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }

    /**
     * @copydoc get_sub_diagonal()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_sub_diagonal() const noexcept
    {
        return sub_diagonal_.get_const_data();
    }

    /**
     * @copydoc get_sub_diagonal(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_sub_diagonal(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return sub_diagonal_.get_const_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }

    /**
     * @copydoc get_main_diagonal()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_main_diagonal() const noexcept
    {
        return main_diagonal_.get_const_data();
    }

    /**
     * @copydoc get_main_diagonal(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_main_diagonal(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return main_diagonal_.get_const_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }

    /**
     * @copydoc get_super_diagonal()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_super_diagonal() const noexcept
    {
        return super_diagonal_.get_const_data();
    }

    /**
     * @copydoc get_super_diagonal(size_type)
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_super_diagonal(size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return super_diagonal_.get_const_data() +
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
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
        return sub_diagonal_.get_num_elems() + main_diagonal_.get_num_elems() +
               super_diagonal_.get_num_elems();
    }

    /**
     * Returns the number of elements stored per diagonal in the batch matrix
     * cumulatively across all batches
     *
     *
     * @return the number of elements stored per diagonal in the batch matrix
     * cumulatively across all batches
     *
     */
    size_type get_num_stored_elements_per_diagonal() const noexcept
    {
        return main_diagonal_.get_num_elems();
    }

    /**
     * Returns the number of elements stored per diagonal for a particular batch
     * entry
     *
     * @return the number of elements stored per diagonal for a particular batch
     * entry
     *
     */
    size_type get_num_stored_elements_per_diagonal(
        const size_type batch) const noexcept
    {
        GKO_ASSERT(batch < this->get_num_batch_entries());
        return num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch + 1] -
               num_elems_per_diagonal_of_the_batch_cumul_
                   .get_const_data()[batch];
    }

    /**
     * Creates a constant (immutable) batch tridiagonal matrix from a constant
     * array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param subdiagonal the sub-diagonal array
     * @param maindiagonal the main-diagonal array
     * @param superdiagonal the super-diagonal array
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const BatchTridiagonal> create_const(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes,
        gko::detail::const_array_view<ValueType>&& subdiagonal,
        gko::detail::const_array_view<ValueType>&& maindiagonal,
        gko::detail::const_array_view<ValueType>&& superdiagonal)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const BatchTridiagonal>(new BatchTridiagonal{
            exec, sizes, gko::detail::array_const_cast(std::move(subdiagonal)),
            gko::detail::array_const_cast(std::move(maindiagonal)),
            gko::detail::array_const_cast(std::move(superdiagonal))});
    }

private:
    /**
     * Compute the memory required for the diagonal(sub/main/super) array from
     * the sizes.
     */
    inline size_type compute_batch_mem_per_diag(const batch_dim<2>& sizes)
    {
        if (sizes.stores_equal_sizes()) {
            return sizes.at(0)[0] * sizes.get_num_batch_entries();
        }
        size_type mem_req = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            mem_req += sizes.at(i)[0];
        }
        return mem_req;
    }

    /**
     * Compute the number of elements stored per diagonal for each batch entry
     * and store it in a prefixed sum fashion
     */
    inline array<size_type> compute_num_elems_per_diagonal_of_the_batch_cumul(
        std::shared_ptr<const Executor> exec, const batch_dim<2>& sizes)
    {
        auto num_elems = array<size_type>(exec->get_master(),
                                          sizes.get_num_batch_entries() + 1);
        num_elems.get_data()[0] = 0;
        for (auto i = 0; i < sizes.get_num_batch_entries(); ++i) {
            num_elems.get_data()[i + 1] =
                num_elems.get_data()[i] + (sizes.at(i))[0];
        }
        num_elems.set_executor(exec);
        return num_elems;
    }

protected:
    /**
     * Creates an uninitialized BatchTridiagonal matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the batch matrices in a batch_dim object
     */
    BatchTridiagonal(std::shared_ptr<const Executor> exec,
                     const batch_dim<2>& size = batch_dim<2>{})
        : EnableBatchLinOp<BatchTridiagonal>(exec, size),
          sub_diagonal_(exec, compute_batch_mem_per_diag(size)),
          main_diagonal_(exec, compute_batch_mem_per_diag(size)),
          super_diagonal_(exec, compute_batch_mem_per_diag(size))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        num_elems_per_diagonal_of_the_batch_cumul_ =
            compute_num_elems_per_diagonal_of_the_batch_cumul(exec,
                                                              this->get_size());
    }

    /**
     * Creates a BatchTridiagonal matrix from an already allocated (and
     * initialized) array.
     *
     * @tparam ValuesArray  type of array of values
     *
     * @param exec  Executor associated to the matrix
     * @param size  sizes of the batch matrices in a batch_dim object
     * @param subdiagonal the sub-diagonal array
     * @param maindiagonal the main-diagonal array
     * @param superdiagonal the super-diagonal array
     *
     * @note If one of `subdiagonal`, `maindiagonal` or `superdiagonal` is not
     * an rvalue, not an array of ValueType or is on the wrong executor, an
     * internal copy of that array will be created, and the original array data
     * will not be used in the matrix.
     *
     */
    template <typename ValuesArray>
    BatchTridiagonal(std::shared_ptr<const Executor> exec,
                     const batch_dim<2>& size, ValuesArray&& subdiagonal,
                     ValuesArray&& maindiagonal, ValuesArray&& superdiagonal)
        : EnableBatchLinOp<BatchTridiagonal>(exec, size),
          sub_diagonal_{exec, std::forward<ValuesArray>(subdiagonal)},
          main_diagonal_{exec, std::forward<ValuesArray>(maindiagonal)},
          super_diagonal_{exec, std::forward<ValuesArray>(superdiagonal)},
          num_elems_per_diagonal_of_the_batch_cumul_(
              exec->get_master(),
              compute_num_elems_per_diagonal_of_the_batch_cumul(
                  exec->get_master(), this->get_size()))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        auto num_elems =
            num_elems_per_diagonal_of_the_batch_cumul_.get_const_data()
                [num_elems_per_diagonal_of_the_batch_cumul_.get_num_elems() -
                 1] -
            1;
        GKO_ENSURE_IN_BOUNDS(num_elems, main_diagonal_.get_num_elems());
        GKO_ASSERT(sub_diagonal_.get_num_elems() ==
                   main_diagonal_.get_num_elems());
        GKO_ASSERT(super_diagonal_.get_num_elems() ==
                   main_diagonal_.get_num_elems());
    }


    /**
     * Creates a BatchTridiagonal matrix by duplicating BatchTridiagonal matrix
     *
     * @param exec  Executor associated to the matrix
     * @param num_duplications  The number of times to duplicate
     * @param input  The matrix to be duplicated.
     */
    // NOTE: Currently, this only works for a uniform batch
    BatchTridiagonal(std::shared_ptr<const Executor> exec,
                     size_type num_duplications,
                     const BatchTridiagonal<value_type>* input)
        : EnableBatchLinOp<BatchTridiagonal>(
              exec, gko::batch_dim<2>(
                        input->get_num_batch_entries() * num_duplications,
                        input->get_size().at(0))),
          sub_diagonal_(exec, compute_batch_mem_per_diag(this->get_size())),
          main_diagonal_(exec, compute_batch_mem_per_diag(this->get_size())),
          super_diagonal_(exec, compute_batch_mem_per_diag(this->get_size()))
    {
        GKO_ASSERT_BATCH_HAS_SQUARE_MATRICES(this);

        num_elems_per_diagonal_of_the_batch_cumul_ =
            compute_num_elems_per_diagonal_of_the_batch_cumul(
                exec->get_master(), this->get_size());
        size_type offset = 0;
        for (size_type i = 0; i < num_duplications; ++i) {
            exec->copy_from(input->get_executor().get(),
                            input->get_num_stored_elements_per_diagonal(),
                            input->get_const_sub_diagonal(),
                            this->get_sub_diagonal() + offset);
            exec->copy_from(input->get_executor().get(),
                            input->get_num_stored_elements_per_diagonal(),
                            input->get_const_main_diagonal(),
                            this->get_main_diagonal() + offset);
            exec->copy_from(input->get_executor().get(),
                            input->get_num_stored_elements_per_diagonal(),
                            input->get_const_super_diagonal(),
                            this->get_super_diagonal() + offset);

            offset += input->get_num_stored_elements_per_diagonal();
        }
    }

    /**
     * Creates a BatchTridiagonal matrix with the same configuration as the
     * callers matrix.
     *
     * @returns a BatchTridiagonal matrix with the same configuration as the
     * caller.
     */
    virtual std::unique_ptr<BatchTridiagonal> create_with_same_config() const
    {
        return BatchTridiagonal::create(this->get_executor(), this->get_size());
    }

    void apply_impl(const BatchLinOp* b, BatchLinOp* x) const override;

    void apply_impl(const BatchLinOp* alpha, const BatchLinOp* b,
                    const BatchLinOp* beta, BatchLinOp* x) const override;

private:
    array<size_type> num_elems_per_diagonal_of_the_batch_cumul_;
    array<value_type> sub_diagonal_;
    array<value_type> main_diagonal_;
    array<value_type> super_diagonal_;

    void add_scaled_identity_impl(const BatchLinOp* a,
                                  const BatchLinOp* b) override;
};


}  // namespace matrix


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_BATCH_TRIDIAGONAL_HPP_
