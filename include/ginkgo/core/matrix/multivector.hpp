// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_DENSE_HPP_
#define GKO_PUBLIC_CORE_MATRIX_DENSE_HPP_


#include <initializer_list>
#include <type_traits>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/multivector_mixin.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/device_views.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>

#include "ginkgo/core/base/multivector_mixin.hpp"


namespace gko {
namespace experimental {
namespace distributed {


template <typename ValueType>
class Vector;


namespace detail {


template <typename ValueType>
class VectorCache;


}  // namespace detail
}  // namespace distributed
}  // namespace experimental


namespace matrix {


template <typename ValueType>
class Dense;


/**
 * MultiVector is a matrix format which explicitly stores all values of the
 * matrix.
 *
 * The values are stored in row-major format (values belonging to the same row
 * appear consecutive in the memory). Optionally, rows can be padded for better
 * memory access.
 *
 * @tparam ValueType  precision of matrix elements
 *
 * @note While this format is not very useful for storing sparse matrices, it
 *       is often suitable to store vectors, and sets of vectors.
 * @ingroup dense
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision>
class MultiVector
    : public EnableMultiVector<MultiVector<ValueType>>,
      public ConvertibleTo<MultiVector<next_precision<ValueType>>>,
#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<MultiVector<next_precision<ValueType, 2>>>,
#endif
#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
      public ConvertibleTo<MultiVector<next_precision<ValueType, 3>>>,
#endif
      public ConvertibleTo<Dense<ValueType>>,
      public ReadableFromMatrixData<ValueType, int32>,
      public ReadableFromMatrixData<ValueType, int64>,
      public WritableToMatrixData<ValueType, int32>,
      public WritableToMatrixData<ValueType, int64> {
    friend class Dense<ValueType>;
    friend class MultiVector<to_complex<ValueType>>;
    friend class EnableMultiVector<MultiVector>;
    friend class experimental::distributed::Vector<ValueType>;
    friend class experimental::distributed::detail::VectorCache<ValueType>;
    GKO_ASSERT_SUPPORTED_VALUE_TYPE;

public:
    using EnableMultiVector<MultiVector>::convert_to;
    using EnableMultiVector<MultiVector>::move_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType>>>::move_to;
    using ConvertibleTo<Dense<ValueType>>::convert_to;
    using ConvertibleTo<Dense<ValueType>>::move_to;
    using ReadableFromMatrixData<ValueType, int32>::read;
    using ReadableFromMatrixData<ValueType, int64>::read;

    using value_type = typename EnableMultiVector<MultiVector>::value_type;
    using index_type = int64;
    using transposed_type = MultiVector<value_type>;
    using mat_data64 = matrix_data<value_type, int64>;
    using mat_data32 = matrix_data<value_type, int32>;
    using device_mat_data64 = device_matrix_data<value_type, int64>;
    using device_mat_data32 = device_matrix_data<value_type, int32>;
    using absolute_type =
        typename EnableMultiVector<MultiVector>::absolute_type;
    using real_type = typename EnableMultiVector<MultiVector>::real_type;
    using complex_type = typename EnableMultiVector<MultiVector>::complex_type;
    using norm_type = typename EnableMultiVector<MultiVector>::norm_type;
    using device_view = typename EnableMultiVector<MultiVector>::device_view;
    using const_device_view =
        typename EnableMultiVector<MultiVector>::const_device_view;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    [[nodiscard]] static std::unique_ptr<MultiVector> create_with_type_of(
        ptr_param<const MultiVector> other,
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type stride);

    friend class MultiVector<previous_precision<ValueType>>;

    void convert_to(
        MultiVector<next_precision<ValueType>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType>>* result) override;

#if GINKGO_ENABLE_HALF || GINKGO_ENABLE_BFLOAT16
    friend class MultiVector<previous_precision<ValueType, 2>>;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 2>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 2>>>::move_to;

    void convert_to(
        MultiVector<next_precision<ValueType, 2>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType, 2>>* result) override;
#endif

#if GINKGO_ENABLE_HALF && GINKGO_ENABLE_BFLOAT16
    friend class MultiVector<previous_precision<ValueType, 3>>;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 3>>>::convert_to;
    using ConvertibleTo<MultiVector<next_precision<ValueType, 3>>>::move_to;

    void convert_to(
        MultiVector<next_precision<ValueType, 3>>* result) const override;

    void move_to(MultiVector<next_precision<ValueType, 3>>* result) override;
#endif

    void convert_to(Dense<ValueType>* result) const override;

    void move_to(Dense<ValueType>* result) override;

    void read(const mat_data64& data) override;

    void read(const mat_data32& data) override;

    void read(const device_mat_data64& data) override;

    void read(const device_mat_data32& data) override;

    void read(device_mat_data64&& data) override;

    void read(device_mat_data32&& data) override;

    void write(mat_data64& data) const override;

    void write(mat_data32& data) const override;

    std::unique_ptr<MultiVector> transpose() const;

    std::unique_ptr<MultiVector> conj_transpose() const;

    /**
     * Writes the transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void transpose(ptr_param<MultiVector> output) const;

    /**
     * Writes the conjugate-transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void conj_transpose(ptr_param<MultiVector> output) const;

    /**
     * Creates a permuted copy \f$A'\f$ of this matrix \f$A\f$ with the given
     * permutation \f$P\f$. By default, this computes a symmetric permutation
     * (permute_mode::symmetric). For the effect of the different permutation
     * modes, see @ref permute_mode.
     *
     * @param permutation  The input permutation.
     * @param mode  The permutation mode, see @ref permute_mode.
     * @return  The permuted matrix.
     */
    std::unique_ptr<MultiVector> permute(
        ptr_param<const Permutation<int32>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, permute_mode)
     */
    std::unique_ptr<MultiVector> permute(
        ptr_param<const Permutation<int64>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Overload of permute(ptr_param<const Permutation<int32>>, permute_mode)
     * that writes the permuted copy into an existing MultiVector.
     * @param output  the output matrix.
     */
    void permute(ptr_param<const Permutation<int32>> permutation,
                 ptr_param<MultiVector> output, permute_mode mode) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>,
     * ptr_param<MultiVector>, permute_mode)
     */
    void permute(ptr_param<const Permutation<int64>> permutation,
                 ptr_param<MultiVector> output, permute_mode mode) const;

    /**
     * Creates a non-symmetrically permuted copy \f$A'\f$ of this matrix \f$A\f$
     * with the given row and column permutations \f$P\f$ and \f$Q\f$. The
     * operation will compute \f$A'(i, j) = A(p[i], q[j])\f$, or \f$A' = P A
     * Q^T\f$ if `invert` is `false`, and \f$A'(p[i], q[j]) = A(i,j)\f$, or
     * \f$A' = P^{-1} A Q^{-T}\f$ if `invert` is `true`.
     *
     * @param row_permutation  The permutation \f$P\f$ to apply to the rows
     * @param column_permutation  The permutation \f$Q\f$ to apply to the
     * columns
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses \f$P^{-1}, Q^{-1}\f$
     * @return  The permuted matrix.
     */
    std::unique_ptr<MultiVector> permute(
        ptr_param<const Permutation<int32>> row_permutation,
        ptr_param<const Permutation<int32>> column_permutation,
        bool invert = false) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, permute_mode)
     */
    std::unique_ptr<MultiVector> permute(
        ptr_param<const Permutation<int64>> row_permutation,
        ptr_param<const Permutation<int64>> column_permutation,
        bool invert = false) const;

    /**
     * Overload of permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, permute_mode) that writes the permuted copy into an
     * existing MultiVector.
     * @param output  the output matrix.
     */
    void permute(ptr_param<const Permutation<int32>> row_permutation,
                 ptr_param<const Permutation<int32>> column_permutation,
                 ptr_param<MultiVector> output, bool invert = false) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, ptr_param<MultiVector>, permute_mode)
     */
    void permute(ptr_param<const Permutation<int64>> row_permutation,
                 ptr_param<const Permutation<int64>> column_permutation,
                 ptr_param<MultiVector> output, bool invert = false) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the permutation modes, see
     * @ref permute(ptr_param<const Permutation<index_type>>, permute_mode)
     *
     * @param permutation  The scaled permutation.
     * @param mode  The permutation mode.
     * @return The permuted matrix.
     */
    std::unique_ptr<MultiVector> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, permute_mode)
     */
    std::unique_ptr<MultiVector> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Overload of scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, permute_mode) that writes the permuted copy into an
     * existing MultiVector.
     * @param output  the output matrix.
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> permutation,
        ptr_param<MultiVector> output, permute_mode mode) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<MultiVector>, permute_mode)
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> permutation,
        ptr_param<MultiVector> output, permute_mode mode) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the parameters, see
     * @ref permute(ptr_param<const Permutation<index_type>>, ptr_param<const
     * Permutation<index_type>>, permute_mode)
     *
     * @param row_permutation  The scaled row permutation.
     * @param column_permutation  The scaled column permutation.
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses \f$P^{-1}, Q^{-1}\f$
     * @return The permuted matrix.
     */
    std::unique_ptr<MultiVector> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int32>>
            column_permutation,
        bool invert = false) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>, bool)
     */
    std::unique_ptr<MultiVector> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int64>>
            column_permutation,
        bool invert = false) const;

    /**
     * Overload of scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>, bool)
     * that writes the permuted copy into an existing MultiVector.
     * @param output  the output matrix.
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int32>>
            column_permutation,
        ptr_param<MultiVector> output, bool invert = false) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>,
     * ptr_param<MultiVector>, bool)
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int64>>
            column_permutation,
        ptr_param<MultiVector> output, bool invert = false) const;

    std::unique_ptr<MultiVector> permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the symmetrically permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::permute(const array<int32>*)
     */
    void permute(const array<int32>* permutation_indices,
                 ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::permute(const array<int32>*, MultiVector*)
     */
    void permute(const array<int64>* permutation_indices,
                 ptr_param<MultiVector> output) const;

    std::unique_ptr<MultiVector> inverse_permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> inverse_permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the inverse symmetrically permuted matrix into the given output
     * matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::inverse_permute(const array<int32>*)
     */
    void inverse_permute(const array<int32>* permutation_indices,
                         ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::inverse_permute(const array<int32>*, MultiVector*)
     */
    void inverse_permute(const array<int64>* permutation_indices,
                         ptr_param<MultiVector> output) const;

    std::unique_ptr<MultiVector> row_permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> row_permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the row-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::row_permute(const array<int32>*)
     */
    void row_permute(const array<int32>* permutation_indices,
                     ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::row_permute(const array<int32>*, MultiVector*)
     */
    void row_permute(const array<int64>* permutation_indices,
                     ptr_param<MultiVector> output) const;

    /**
     * Create a MultiVector consisting of the given rows from this
     * matrix.
     *
     * @param gather_indices  pointer to an array containing row indices
     *                        from this matrix. It may contain duplicates.
     * @return  MultiVector on the same executor with the same number of
     *          columns and `gather_indices->get_size()` rows containing
     *          the gathered rows from this matrix:
     *          `output(i,j) = input(gather_indices(i), j)`
     */
    std::unique_ptr<MultiVector> row_gather(
        const array<int32>* gather_indices) const;

    /**
     * @copydoc row_gather(const array<int32>*) const
     */
    std::unique_ptr<MultiVector> row_gather(
        const array<int64>* gather_indices) const;

    /**
     * Copies the given rows from this matrix into `row_collection`
     *
     * @param gather_indices  pointer to an array containing row indices
     *                        from this matrix. It may contain duplicates.
     * @param row_collection  pointer to a LinOp that will store the gathered
     *                        rows:
     *                        `row_collection(i,j)
     *                         = input(gather_indices(i), j)`
     *                        It must have the same number of columns as this
     *                        matrix and `gather_indices->get_size()` rows.
     */
    void row_gather(const array<int32>* gather_indices,
                    ptr_param<AbstractMultiVector> row_collection) const;

    /**
     * @copydoc row_gather(const array<int32>*, LinOp*) const
     */
    void row_gather(const array<int64>* gather_indices,
                    ptr_param<AbstractMultiVector> row_collection) const;

    /**
     * Copies the given rows from this matrix into `row_collection` with scaling
     *
     * @param alpha  scaling the result of row gathering
     * @param gather_indices  pointer to an array containing row indices
     *                        from this matrix. It may contain duplicates.
     * @param beta  scaling the input row_collection
     * @param row_collection  pointer to a LinOp that will store the
     *             gathered rows:
     *             `row_collection(i,j) = input(gather_indices(i), j)`
     *             It must have the same number of columns as this
     *             matrix and `gather_indices->get_size()` rows.
     */
    void row_gather(ptr_param<const LinOp> alpha,
                    const array<int32>* gather_indices,
                    ptr_param<const LinOp> beta,
                    ptr_param<AbstractMultiVector> row_collection) const;

    /**
     * @copydoc row_gather(const LinOp*, const array<int32>*, const LinOp*,
     * LinOp*) const
     */
    void row_gather(ptr_param<const LinOp> alpha,
                    const array<int64>* gather_indices,
                    ptr_param<const LinOp> beta,
                    ptr_param<AbstractMultiVector> row_collection) const;

    std::unique_ptr<MultiVector> column_permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> column_permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the column-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[1]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::column_permute(const array<int32>*)
     */
    void column_permute(const array<int32>* permutation_indices,
                        ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::column_permute(const array<int32>*, MultiVector*)
     */
    void column_permute(const array<int64>* permutation_indices,
                        ptr_param<MultiVector> output) const;

    std::unique_ptr<MultiVector> inverse_row_permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> inverse_row_permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the inverse row-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::inverse_row_permute(const array<int32>*)
     */
    void inverse_row_permute(const array<int32>* permutation_indices,
                             ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::inverse_row_permute(const array<int32>*,
     * MultiVector*)
     */
    void inverse_row_permute(const array<int64>* permutation_indices,
                             ptr_param<MultiVector> output) const;

    std::unique_ptr<MultiVector> inverse_column_permute(
        const array<int32>* permutation_indices) const;

    std::unique_ptr<MultiVector> inverse_column_permute(
        const array<int64>* permutation_indices) const;

    /**
     * Writes the inverse column-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[1]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see MultiVector::inverse_column_permute(const array<int32>*)
     */
    void inverse_column_permute(const array<int32>* permutation_indices,
                                ptr_param<MultiVector> output) const;

    /**
     * @copydoc MultiVector::inverse_column_permute(const array<int32>*,
     * MultiVector*)
     */
    void inverse_column_permute(const array<int64>* permutation_indices,
                                ptr_param<MultiVector> output) const;

    /**
     * Returns a pointer to the array of values of the matrix.
     *
     * @return the pointer to the array of values
     */
    value_type* get_values() noexcept { return values_.get_data(); }

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
        return values_.get_size();
    }

    device_view get_device_view();

    const_device_view get_const_device_view() const;

    /**
     * Returns a single element of the matrix.
     *
     * @param row  the row of the requested element
     * @param col  the column of the requested element
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    value_type& at(size_type row, size_type col) noexcept
    {
        return values_.get_data()[linearize_index(row, col)];
    }

    /**
     * @copydoc MultiVector::at(size_type, size_type)
     */
    value_type at(size_type row, size_type col) const noexcept
    {
        return values_.get_const_data()[linearize_index(row, col)];
    }

    /**
     * Returns a single element of the matrix.
     *
     * Useful for iterating across all elements of the matrix.
     * However, it is less efficient than the two-parameter variant of this
     * method.
     *
     * @param idx  a linear index of the requested element
     *             (ignoring the stride)
     *
     * @note  the method has to be called on the same Executor the matrix is
     *        stored at (e.g. trying to call this method on a GPU matrix from
     *        the OMP results in a runtime error)
     */
    ValueType& at(size_type idx) noexcept
    {
        return values_.get_data()[linearize_index(idx)];
    }

    /**
     * @copydoc MultiVector::at(size_type)
     */
    ValueType at(size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(idx)];
    }

    /**
     * Computes the column-wise arithmetic mean of this matrix.
     *
     * @param result  a MultiVector row vector, used to store the mean
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_mean(ptr_param<AbstractMultiVector> result) const;

    /**
     * Computes the column-wise arithmetic mean of this matrix.
     *
     * @param result  a MultiVector row vector, used to store the mean
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_mean(ptr_param<AbstractMultiVector> result,
                      array<char>& tmp) const;

    void validate_data() const override;

    /**
     * Creates an uninitialized MultiVector of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements).
     *                  If it is set to 0, size[1] will be used instead.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size = {},
        size_type stride = 0);

    /**
     * Creates a MultiVector from an already allocated (and initialized)
     * array.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param values  array of matrix values
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     *
     * @note If `values` is not an rvalue, not an array of ValueType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     *
     * @return A smart pointer to the newly created matrix.
     */
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<value_type> values, size_type stride);

    /**
     * @copydoc std::unique_ptr<MultiVector> create(std::shared_ptr<const
     * Executor>, const dim<2>&, array<value_type>, size_type)
     */
    template <typename InputValueType>
    GKO_DEPRECATED(
        "explicitly construct the gko::array argument instead of passing an"
        "initializer list")
    static std::unique_ptr<MultiVector> create(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        std::initializer_list<InputValueType> values, size_type stride)
    {
        return create(exec, size, array<value_type>{exec, std::move(values)},
                      stride);
    }

    /**
     * Creates a constant (immutable) MultiVector from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param stride  the row-stride of the matrix
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const MultiVector> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values, size_type stride);

    [[nodiscard]] std::unique_ptr<const Dense<ValueType>> as_const_dense_view()
        const;

    [[nodiscard]] std::unique_ptr<Dense<ValueType>> as_dense_view();

    /**
     * Converts the vector to the target precision type.
     *
     * @note This overload will include a copy-back operation when the temporary
     *       conversion is destroyed, if OtherValueType != ValueType.
     *
     * @tparam OtherValueType The target precision type. If ValueType is real,
     *                         OtherValueType must be real. If ValueType is
     *                         complex, OtherValueType must be complex.
     *
     * @return Temporary conversion to the target precision type.
     */
    template <typename OtherValueType,
              typename = std::enable_if_t<is_complex<ValueType>() ==
                                          is_complex<OtherValueType>()>>
    [[nodiscard]] temporary_conversion<MultiVector<OtherValueType>>
    as_precision();

    /**
     * Converts the vector to the target precision type.
     *
     * @tparam OtherValueType The target precision type. If ValueType is real,
     *                         OtherValueType must be real. If ValueType is
     *                         complex, OtherValueType must be complex.
     *
     * @return Temporary conversion to the target precision type.
     */
    template <typename OtherValueType,
              typename = std::enable_if_t<is_complex<ValueType>() ==
                                          is_complex<OtherValueType>()>>
    [[nodiscard]] temporary_conversion<const MultiVector<OtherValueType>>
    as_precision() const;

    /**
     * Copy-assigns a MultiVector. Preserves the executor, reallocates
     * the matrix with minimal stride if the dimensions don't match, then copies
     * the data over, ignoring padding.
     */
    MultiVector& operator=(const MultiVector&);

    /**
     * Move-assigns a MultiVector. Preserves the executor, moves the data
     * over preserving size and stride. Leaves the moved-from object in an empty
     * state (0x0 with empty Array).
     */
    MultiVector& operator=(MultiVector&&);

    /**
     * Copy-constructs a MultiVector. Inherits executor and dimensions,
     * but copies data without padding.
     */
    MultiVector(const MultiVector&);

    /**
     * Move-constructs a MultiVector. Inherits executor, dimensions and
     * data with padding. The moved-from object is empty (0x0 with empty Array).
     */
    MultiVector(MultiVector&&);

protected:
    MultiVector(std::shared_ptr<const Executor> exec, const dim<2>& size = {},
                size_type stride = 0);

    MultiVector(std::shared_ptr<const Executor> exec, const dim<2>& size,
                array<value_type> values, size_type stride);

    /**
     * Creates a MultiVector with the same type as the callers matrix.
     *
     * @param size  size of the matrix
     *
     * @returns a MultiVector with the same type as the caller.
     */
    virtual std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type stride) const
    {
        return MultiVector::create(exec, size, stride);
    }

    /**
     * @copydoc compute_mean(LinOp*) const
     */
    virtual void compute_mean_impl(AbstractMultiVector* result) const;

    /**
     * Resizes the matrix to the given size.
     *
     * If the new size matches the current size, the stride will be left
     * unchanged, otherwise it will be set to the number of columns.
     *
     * @param new_size  the new matrix dimensions
     */
    void resize(gko::dim<2> new_size);

    size_type linearize_index(size_type row, size_type col) const noexcept
    {
        return row * stride_ + col;
    }

    size_type linearize_index(size_type idx) const noexcept
    {
        return linearize_index(idx / this->get_size()[1],
                               idx % this->get_size()[1]);
    }

    template <typename IndexType>
    void permute_impl(const Permutation<IndexType>* permutation,
                      permute_mode mode, MultiVector* output) const;

    template <typename IndexType>
    void permute_impl(const Permutation<IndexType>* row_permutation,
                      const Permutation<IndexType>* col_permutation,
                      bool invert, MultiVector* output) const;

    template <typename IndexType>
    void scale_permute_impl(
        const ScaledPermutation<ValueType, IndexType>* permutation,
        permute_mode mode, MultiVector* output) const;

    template <typename IndexType>
    void scale_permute_impl(
        const ScaledPermutation<ValueType, IndexType>* row_permutation,
        const ScaledPermutation<ValueType, IndexType>* column_permutation,
        bool invert, MultiVector* output) const;

    template <typename OutputType, typename IndexType>
    void row_gather_impl(const array<IndexType>* row_idxs,
                         MultiVector<OutputType>* row_collection) const;

    template <typename OutputType, typename IndexType>
    void row_gather_impl(const MultiVector<ValueType>* alpha,
                         const array<IndexType>* row_idxs,
                         const MultiVector<ValueType>* beta,
                         MultiVector<OutputType>* row_collection) const;

    void compute_absolute_inplace_impl() override;

    [[nodiscard]] std::unique_ptr<MultiVector> create_with_same_config_impl()
        const override;

    [[nodiscard]] std::unique_ptr<MultiVector> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& global_size,
        const dim<2>& local_size, size_type stride) const override;

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview_impl(
        local_span rows, local_span columns) override;

    [[nodiscard]] std::unique_ptr<const MultiVector> create_subview_impl(
        local_span rows, local_span columns) const override;

    [[nodiscard]] std::unique_ptr<MultiVector> create_subview_impl(
        local_span rows, local_span columns, dim<2> global_size) override;

    [[nodiscard]] std::unique_ptr<const MultiVector> create_subview_impl(
        local_span rows, local_span columns, dim<2> global_size) const override;

    [[nodiscard]] std::unique_ptr<const real_type> create_real_view_impl()
        const override;

    [[nodiscard]] std::unique_ptr<real_type> create_real_view_impl() override;

    [[nodiscard]] std::unique_ptr<absolute_type> compute_absolute_impl()
        const override;

    void compute_absolute_impl(absolute_type* result) const override;

    [[nodiscard]] std::unique_ptr<complex_type> make_complex_impl()
        const override;

    [[nodiscard]] std::unique_ptr<real_type> get_real_impl() const override;

    [[nodiscard]] std::unique_ptr<real_type> get_imag_impl() const override;

    void make_complex_impl(complex_type* result) const override;

    void get_real_impl(real_type* result) const override;

    void get_imag_impl(real_type* result) const override;

    void fill_impl(value_type value) override;

    void scale_impl(scaling_param<value_type> alpha) override;

    void inv_scale_impl(scaling_param<value_type> alpha) override;

    void add_scaled_impl(scaling_param<value_type> alpha,
                         const MultiVector* b) override;

    void sub_scaled_impl(scaling_param<value_type> alpha,
                         const MultiVector* b) override;

    void compute_dot_impl(const MultiVector* b,
                          matrix::MultiVector<value_type>* result,
                          array<char>& tmp) const override;

    void compute_conj_dot_impl(const MultiVector* b,
                               matrix::MultiVector<value_type>* result,
                               array<char>& tmp) const override;

    void compute_norm2_impl(norm_type* result, array<char>& tmp) const override;

    void compute_squared_norm2_impl(norm_type* result,
                                    array<char>& tmp) const override;

    void compute_norm1_impl(norm_type* result, array<char>& tmp) const override;
    AbstractMultiVector::device_view<value_type> get_local_device_view_impl()
        override;

    AbstractMultiVector::device_view<const value_type>
    get_const_local_device_view_impl() const override;

private:
    size_type stride_;
    array<value_type> values_;
};


}  // namespace matrix


namespace detail {


template <typename ValueType>
struct temporary_clone_helper<matrix::MultiVector<ValueType>> {
    static std::unique_ptr<matrix::MultiVector<ValueType>> create(
        std::shared_ptr<const Executor> exec,
        matrix::MultiVector<ValueType>* ptr, bool copy_data)
    {
        if (copy_data) {
            return gko::clone(std::move(exec), ptr);
        } else {
            return matrix::MultiVector<ValueType>::create(exec,
                                                          ptr->get_size());
        }
    }
};


}  // namespace detail


/**
 * Creates a view of a given MultiVector vector.
 *
 * @tparam VecPtr  a (smart or raw) pointer to the vector.
 *
 * @param vector  the vector on which to create the view
 */
template <typename VecPtr>
std::unique_ptr<
    matrix::MultiVector<typename detail::pointee<VecPtr>::value_type>>
make_dense_view(VecPtr&& vector)
{
    return vector->create_subview({0, vector->get_size()[0]},
                                  {0, vector->get_size()[1]});
}


/**
 * Creates a view of a given MultiVector vector.
 *
 * @tparam VecPtr  a (smart or raw) pointer to the vector.
 *
 * @param vector  the vector on which to create the view
 */
template <typename VecPtr>
std::unique_ptr<
    const matrix::MultiVector<typename detail::pointee<VecPtr>::value_type>>
make_const_dense_view(VecPtr&& vector)
{
    return vector->create_subview({0, vector->get_size()[0]},
                                  {0, vector->get_size()[1]});
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DENSE_HPP_
