// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
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
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/permutation.hpp>
#include <ginkgo/core/matrix/scaled_permutation.hpp>


namespace gko {
namespace experimental {
namespace distributed {


template <typename ValueType>
class Vector;


}
}  // namespace experimental


namespace matrix {


template <typename ValueType, typename IndexType>
class Coo;

template <typename ValueType, typename IndexType>
class Csr;

template <typename ValueType>
class Diagonal;

template <typename ValueType, typename IndexType>
class Ell;

template <typename ValueType, typename IndexType>
class Fbcsr;

template <typename ValueType, typename IndexType>
class Hybrid;

template <typename ValueType, typename IndexType>
class Sellp;

template <typename ValueType, typename IndexType>
class SparsityCsr;


/**
 * Dense is a matrix format which explicitly stores all values of the matrix.
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
class Dense
    : public EnableLinOp<Dense<ValueType>>,
      public EnableCreateMethod<Dense<ValueType>>,
      public ConvertibleTo<Dense<next_precision<ValueType>>>,
      public ConvertibleTo<Coo<ValueType, int32>>,
      public ConvertibleTo<Coo<ValueType, int64>>,
      public ConvertibleTo<Csr<ValueType, int32>>,
      public ConvertibleTo<Csr<ValueType, int64>>,
      public ConvertibleTo<Ell<ValueType, int32>>,
      public ConvertibleTo<Ell<ValueType, int64>>,
      public ConvertibleTo<Fbcsr<ValueType, int32>>,
      public ConvertibleTo<Fbcsr<ValueType, int64>>,
      public ConvertibleTo<Hybrid<ValueType, int32>>,
      public ConvertibleTo<Hybrid<ValueType, int64>>,
      public ConvertibleTo<Sellp<ValueType, int32>>,
      public ConvertibleTo<Sellp<ValueType, int64>>,
      public ConvertibleTo<SparsityCsr<ValueType, int32>>,
      public ConvertibleTo<SparsityCsr<ValueType, int64>>,
      public DiagonalExtractable<ValueType>,
      public ReadableFromMatrixData<ValueType, int32>,
      public ReadableFromMatrixData<ValueType, int64>,
      public WritableToMatrixData<ValueType, int32>,
      public WritableToMatrixData<ValueType, int64>,
      public Transposable,
      public Permutable<int32>,
      public Permutable<int64>,
      public EnableAbsoluteComputation<remove_complex<Dense<ValueType>>>,
      public ScaledIdentityAddable {
    friend class EnableCreateMethod<Dense>;
    friend class EnablePolymorphicObject<Dense, LinOp>;
    friend class Coo<ValueType, int32>;
    friend class Coo<ValueType, int64>;
    friend class Csr<ValueType, int32>;
    friend class Csr<ValueType, int64>;
    friend class Diagonal<ValueType>;
    friend class Ell<ValueType, int32>;
    friend class Ell<ValueType, int64>;
    friend class Fbcsr<ValueType, int32>;
    friend class Fbcsr<ValueType, int64>;
    friend class Hybrid<ValueType, int32>;
    friend class Hybrid<ValueType, int64>;
    friend class Sellp<ValueType, int32>;
    friend class Sellp<ValueType, int64>;
    friend class SparsityCsr<ValueType, int32>;
    friend class SparsityCsr<ValueType, int64>;
    friend class Dense<to_complex<ValueType>>;
    friend class experimental::distributed::Vector<ValueType>;

public:
    using EnableLinOp<Dense>::convert_to;
    using EnableLinOp<Dense>::move_to;
    using ConvertibleTo<Dense<next_precision<ValueType>>>::convert_to;
    using ConvertibleTo<Dense<next_precision<ValueType>>>::move_to;
    using ConvertibleTo<Coo<ValueType, int32>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int32>>::move_to;
    using ConvertibleTo<Coo<ValueType, int64>>::convert_to;
    using ConvertibleTo<Coo<ValueType, int64>>::move_to;
    using ConvertibleTo<Csr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int32>>::move_to;
    using ConvertibleTo<Csr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Csr<ValueType, int64>>::move_to;
    using ConvertibleTo<Ell<ValueType, int32>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int32>>::move_to;
    using ConvertibleTo<Ell<ValueType, int64>>::convert_to;
    using ConvertibleTo<Ell<ValueType, int64>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int32>>::move_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<Fbcsr<ValueType, int64>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int32>>::move_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::convert_to;
    using ConvertibleTo<Hybrid<ValueType, int64>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int32>>::move_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::convert_to;
    using ConvertibleTo<Sellp<ValueType, int64>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int32>>::move_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::convert_to;
    using ConvertibleTo<SparsityCsr<ValueType, int64>>::move_to;
    using ReadableFromMatrixData<ValueType, int32>::read;
    using ReadableFromMatrixData<ValueType, int64>::read;

    using value_type = ValueType;
    using index_type = int64;
    using transposed_type = Dense<ValueType>;
    using mat_data = matrix_data<ValueType, int64>;
    using mat_data32 = matrix_data<ValueType, int32>;
    using device_mat_data = device_matrix_data<ValueType, int64>;
    using device_mat_data32 = device_matrix_data<ValueType, int32>;
    using absolute_type = remove_complex<Dense>;
    using real_type = absolute_type;
    using complex_type = to_complex<Dense>;

    using row_major_range = gko::range<gko::accessor::row_major<ValueType, 2>>;

    /**
     * Creates a Dense matrix with the same size and stride as another Dense
     * matrix.
     *
     * @param other  The other matrix whose configuration needs to copied.
     */
    static std::unique_ptr<Dense> create_with_config_of(
        ptr_param<const Dense> other)
    {
        // De-referencing `other` before calling the functions (instead of
        // using operator `->`) is currently required to be compatible with
        // CUDA 10.1.
        // Otherwise, it results in a compile error.
        return (*other).create_with_same_config();
    }

    /**
     * Creates a Dense matrix with the same type as another Dense
     * matrix but on a different executor and with a different size.
     *
     * @param other  The other matrix whose type we target.
     * @param exec  The executor of the new matrix.
     * @param size  The size of the new matrix.
     * @param stride  The stride of the new matrix.
     *
     * @returns a Dense matrix with the type of other.
     */
    static std::unique_ptr<Dense> create_with_type_of(
        ptr_param<const Dense> other, std::shared_ptr<const Executor> exec,
        const dim<2>& size = dim<2>{})
    {
        // See create_with_config_of()
        return (*other).create_with_type_of_impl(exec, size, size[1]);
    }

    /**
     * @copydoc create_with_type_of(const Dense*, std::shared_ptr<const
     * Executor>, const dim<2>)
     *
     * @param stride  The stride of the new matrix.
     *
     * @note This is an overload which allows full parameter specification.
     */
    static std::unique_ptr<Dense> create_with_type_of(
        ptr_param<const Dense> other, std::shared_ptr<const Executor> exec,
        const dim<2>& size, size_type stride)
    {
        // See create_with_config_of()
        return (*other).create_with_type_of_impl(exec, size, stride);
    }

    /**
     * @copydoc create_with_type_of(const Dense*, std::shared_ptr<const
     * Executor>, const dim<2>)
     *
     * @param local_size  Unused
     * @param stride  The stride of the new matrix.
     *
     * @note This is an overload to stay consistent with
     *       gko::experimental::distributed::Vector
     */
    static std::unique_ptr<Dense> create_with_type_of(
        ptr_param<const Dense> other, std::shared_ptr<const Executor> exec,
        const dim<2>& size, const dim<2>& local_size, size_type stride)
    {
        // See create_with_config_of()
        return (*other).create_with_type_of_impl(exec, size, stride);
    }

    /**
     * Creates a Dense matrix, where the underlying array is a view of another
     * Dense matrix' array.
     *
     * @param other  The other matrix on which to create the view
     *
     * @return  A Dense matrix that is a view of other
     */
    static std::unique_ptr<Dense> create_view_of(ptr_param<Dense> other)
    {
        return other->create_view_of_impl();
    }

    /**
     * Creates a immutable Dense matrix, where the underlying array is a view of
     * another Dense matrix' array.
     *
     * @param other  The other matrix on which to create the view
     * @return  A immutable Dense matrix that is a view of other
     */
    static std::unique_ptr<const Dense> create_const_view_of(
        ptr_param<const Dense> other)
    {
        return other->create_const_view_of_impl();
    }

    friend class Dense<next_precision<ValueType>>;

    void convert_to(Dense<next_precision<ValueType>>* result) const override;

    void move_to(Dense<next_precision<ValueType>>* result) override;

    void convert_to(Coo<ValueType, int32>* result) const override;

    void move_to(Coo<ValueType, int32>* result) override;

    void convert_to(Coo<ValueType, int64>* result) const override;

    void move_to(Coo<ValueType, int64>* result) override;

    void convert_to(Csr<ValueType, int32>* result) const override;

    void move_to(Csr<ValueType, int32>* result) override;

    void convert_to(Csr<ValueType, int64>* result) const override;

    void move_to(Csr<ValueType, int64>* result) override;

    void convert_to(Ell<ValueType, int32>* result) const override;

    void move_to(Ell<ValueType, int32>* result) override;

    void convert_to(Ell<ValueType, int64>* result) const override;

    void move_to(Ell<ValueType, int64>* result) override;

    void convert_to(Fbcsr<ValueType, int32>* result) const override;

    void move_to(Fbcsr<ValueType, int32>* result) override;

    void convert_to(Fbcsr<ValueType, int64>* result) const override;

    void move_to(Fbcsr<ValueType, int64>* result) override;

    void convert_to(Hybrid<ValueType, int32>* result) const override;

    void move_to(Hybrid<ValueType, int32>* result) override;

    void convert_to(Hybrid<ValueType, int64>* result) const override;

    void move_to(Hybrid<ValueType, int64>* result) override;

    void convert_to(Sellp<ValueType, int32>* result) const override;

    void move_to(Sellp<ValueType, int32>* result) override;

    void convert_to(Sellp<ValueType, int64>* result) const override;

    void move_to(Sellp<ValueType, int64>* result) override;

    void convert_to(SparsityCsr<ValueType, int32>* result) const override;

    void move_to(SparsityCsr<ValueType, int32>* result) override;

    void convert_to(SparsityCsr<ValueType, int64>* result) const override;

    void move_to(SparsityCsr<ValueType, int64>* result) override;

    void read(const mat_data& data) override;

    void read(const mat_data32& data) override;

    void read(const device_mat_data& data) override;

    void read(const device_mat_data32& data) override;

    void read(device_mat_data&& data) override;

    void read(device_mat_data32&& data) override;

    void write(mat_data& data) const override;

    void write(mat_data32& data) const override;

    std::unique_ptr<LinOp> transpose() const override;

    std::unique_ptr<LinOp> conj_transpose() const override;

    /**
     * Writes the transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void transpose(ptr_param<Dense> output) const;

    /**
     * Writes the conjugate-transposed matrix into the given output matrix.
     *
     * @param output  The output matrix. It must have the dimensions
     *                `gko::transpose(this->get_size())`
     */
    void conj_transpose(ptr_param<Dense> output) const;

    /**
     * Fill the dense matrix with a given value.
     *
     * @param value  the value to be filled
     */
    void fill(const ValueType value);

    /**
     * Creates a permuted copy $A'$ of this matrix $A$ with the given
     * permutation $P$. By default, this computes a symmetric permutation
     * (permute_mode::symmetric). For the effect of the different permutation
     * modes, see @ref permute_mode.
     *
     * @param permutation  The input permutation.
     * @param mode  The permutation mode. If permute_mode::inverse is set, we
     *              use the inverse permutation $P^{-1}$ instead of $P$.
     *              If permute_mode::rows is set, the rows will be permuted.
     *              If permute_mode::columns is set, the columns will be
     *              permuted.
     * @return  The permuted matrix.
     */
    std::unique_ptr<Dense> permute(
        ptr_param<const Permutation<int32>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, permute_mode)
     */
    std::unique_ptr<Dense> permute(
        ptr_param<const Permutation<int64>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Overload of permute(ptr_param<const Permutation<int32>>, permute_mode)
     * that writes the permuted copy into an existing Dense matrix.
     * @param output  the output matrix.
     */
    void permute(ptr_param<const Permutation<int32>> permutation,
                 ptr_param<Dense> output, permute_mode mode) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, ptr_param<Dense>,
     * permute_mode)
     */
    void permute(ptr_param<const Permutation<int64>> permutation,
                 ptr_param<Dense> output, permute_mode mode) const;

    /**
     * Creates a non-symmetrically permuted copy $A'$ of this matrix $A$ with
     * the given row and column permutations $P$ and $Q$. The operation will
     * compute $A'(i, j) = A(p[i], q[j])$, or $A' = P A Q^T$ if `invert` is
     * `false`, and $A'(p[i], q[j]) = A(i,j)$, or $A' = P^{-1} A Q^{-T}$ if
     * `invert` is `true`.
     *
     * @param row_permutation  The permutation $P$ to apply to the rows
     * @param column_permutation  The permutation $Q$ to apply to the columns
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses $P^{-1}, Q^{-1}$
     * @return  The permuted matrix.
     */
    std::unique_ptr<Dense> permute(
        ptr_param<const Permutation<int32>> row_permutation,
        ptr_param<const Permutation<int32>> column_permutation,
        bool invert = false) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, permute_mode)
     */
    std::unique_ptr<Dense> permute(
        ptr_param<const Permutation<int64>> row_permutation,
        ptr_param<const Permutation<int64>> column_permutation,
        bool invert = false) const;

    /**
     * Overload of permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, permute_mode) that writes the permuted copy into an
     * existing Dense matrix.
     * @param output  the output matrix.
     */
    void permute(ptr_param<const Permutation<int32>> row_permutation,
                 ptr_param<const Permutation<int32>> column_permutation,
                 ptr_param<Dense> output, bool invert = false) const;

    /**
     * @copydoc permute(ptr_param<const Permutation<int32>>, ptr_param<const
     * Permutation<int32>>, ptr_param<Dense>, permute_mode)
     */
    void permute(ptr_param<const Permutation<int64>> row_permutation,
                 ptr_param<const Permutation<int64>> column_permutation,
                 ptr_param<Dense> output, bool invert = false) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the permutation modes, see
     * @ref permute(ptr_param<const Permutation<index_type>>, permute_mode)
     *
     * @param permutation  The scaled permutation.
     * @param mode  The permutation mode.
     * @return The permuted matrix.
     */
    std::unique_ptr<Dense> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, permute_mode)
     */
    std::unique_ptr<Dense> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> permutation,
        permute_mode mode = permute_mode::symmetric) const;

    /**
     * Overload of scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, permute_mode) that writes the permuted copy into an
     * existing Dense matrix.
     * @param output  the output matrix.
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> permutation,
        ptr_param<Dense> output, permute_mode mode) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<Dense>, permute_mode)
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> permutation,
        ptr_param<Dense> output, permute_mode mode) const;

    /**
     * Creates a scaled and permuted copy of this matrix.
     * For an explanation of the parameters, see
     * @ref permute(ptr_param<const Permutation<index_type>>, ptr_param<const
     * Permutation<index_type>>, permute_mode)
     *
     * @param row_permutation  The scaled row permutation.
     * @param column_permutation  The scaled column permutation.
     * @param invert  If set to `false`, uses the input permutations, otherwise
     *                uses their inverses $P^{-1}, Q^{-1}$
     * @return The permuted matrix.
     */
    std::unique_ptr<Dense> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int32>>
            column_permutation,
        bool invert = false) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>, bool)
     */
    std::unique_ptr<Dense> scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int64>>
            column_permutation,
        bool invert = false) const;

    /**
     * Overload of scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>, bool)
     * that writes the permuted copy into an existing Dense matrix.
     * @param output  the output matrix.
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int32>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int32>>
            column_permutation,
        ptr_param<Dense> output, bool invert = false) const;

    /**
     * @copydoc scale_permute(ptr_param<const ScaledPermutation<value_type,
     * int32>>, ptr_param<const ScaledPermutation<value_type, int32>>,
     * ptr_param<Dense>, bool)
     */
    void scale_permute(
        ptr_param<const ScaledPermutation<value_type, int64>> row_permutation,
        ptr_param<const ScaledPermutation<value_type, int64>>
            column_permutation,
        ptr_param<Dense> output, bool invert = false) const;

    std::unique_ptr<LinOp> permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the symmetrically permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::permute(const array<int32>*)
     */
    void permute(const array<int32>* permutation_indices,
                 ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::permute(const array<int32>*, Dense*)
     */
    void permute(const array<int64>* permutation_indices,
                 ptr_param<Dense> output) const;

    std::unique_ptr<LinOp> inverse_permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the inverse symmetrically permuted matrix into the given output
     * matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::inverse_permute(const array<int32>*)
     */
    void inverse_permute(const array<int32>* permutation_indices,
                         ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::inverse_permute(const array<int32>*, Dense*)
     */
    void inverse_permute(const array<int64>* permutation_indices,
                         ptr_param<Dense> output) const;

    std::unique_ptr<LinOp> row_permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> row_permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the row-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::row_permute(const array<int32>*)
     */
    void row_permute(const array<int32>* permutation_indices,
                     ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::row_permute(const array<int32>*, Dense*)
     */
    void row_permute(const array<int64>* permutation_indices,
                     ptr_param<Dense> output) const;

    /**
     * Create a Dense matrix consisting of the given rows from this matrix.
     *
     * @param gather_indices  pointer to an array containing row indices
     *                        from this matrix. It may contain duplicates.
     * @return  Dense matrix on the same executor with the same number of
     *          columns and `gather_indices->get_size()` rows containing
     *          the gathered rows from this matrix:
     *          `output(i,j) = input(gather_indices(i), j)`
     */
    std::unique_ptr<Dense> row_gather(const array<int32>* gather_indices) const;

    /**
     * @copydoc row_gather(const array<int32>*) const
     */
    std::unique_ptr<Dense> row_gather(const array<int64>* gather_indices) const;

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
                    ptr_param<LinOp> row_collection) const;

    /**
     * @copydoc row_gather(const array<int32>*, LinOp*) const
     */
    void row_gather(const array<int64>* gather_indices,
                    ptr_param<LinOp> row_collection) const;

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
                    ptr_param<LinOp> row_collection) const;

    /**
     * @copydoc row_gather(const LinOp*, const array<int32>*, const LinOp*,
     * LinOp*) const
     */
    void row_gather(ptr_param<const LinOp> alpha,
                    const array<int64>* gather_indices,
                    ptr_param<const LinOp> beta,
                    ptr_param<LinOp> row_collection) const;

    std::unique_ptr<LinOp> column_permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> column_permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the column-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[1]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::column_permute(const array<int32>*)
     */
    void column_permute(const array<int32>* permutation_indices,
                        ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::column_permute(const array<int32>*, Dense*)
     */
    void column_permute(const array<int64>* permutation_indices,
                        ptr_param<Dense> output) const;

    std::unique_ptr<LinOp> inverse_row_permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_row_permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the inverse row-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[0]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::inverse_row_permute(const array<int32>*)
     */
    void inverse_row_permute(const array<int32>* permutation_indices,
                             ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::inverse_row_permute(const array<int32>*, Dense*)
     */
    void inverse_row_permute(const array<int64>* permutation_indices,
                             ptr_param<Dense> output) const;

    std::unique_ptr<LinOp> inverse_column_permute(
        const array<int32>* permutation_indices) const override;

    std::unique_ptr<LinOp> inverse_column_permute(
        const array<int64>* permutation_indices) const override;

    /**
     * Writes the inverse column-permuted matrix into the given output matrix.
     *
     * @param permutation_indices  The array containing permutation indices.
     *                             It must have `this->get_size()[1]` elements.
     * @param output  The output matrix. It must have the dimensions
     *                `this->get_size()`
     * @see Dense::inverse_column_permute(const array<int32>*)
     */
    void inverse_column_permute(const array<int32>* permutation_indices,
                                ptr_param<Dense> output) const;

    /**
     * @copydoc Dense::inverse_column_permute(const array<int32>*, Dense*)
     */
    void inverse_column_permute(const array<int64>* permutation_indices,
                                ptr_param<Dense> output) const;

    std::unique_ptr<Diagonal<ValueType>> extract_diagonal() const override;

    /**
     * Writes the diagonal of this matrix into an existing diagonal matrix.
     *
     * @param output  The output matrix. Its size must match the size of this
     *                matrix's diagonal.
     * @see Dense::extract_diagonal()
     */
    void extract_diagonal(ptr_param<Diagonal<ValueType>> output) const;

    std::unique_ptr<absolute_type> compute_absolute() const override;

    /**
     * Writes the absolute values of this matrix into an existing matrix.
     *
     * @param output  The output matrix. Its size must match the size of this
     *                matrix.
     * @see Dense::compute_absolute()
     */
    void compute_absolute(ptr_param<absolute_type> output) const;

    void compute_absolute_inplace() override;

    /**
     * Creates a complex copy of the original matrix. If the original matrix
     * was real, the imaginary part of the result will be zero.
     */
    std::unique_ptr<complex_type> make_complex() const;

    /**
     * Writes a complex copy of the original matrix to a given complex matrix.
     * If the original matrix was real, the imaginary part of the result will
     * be zero.
     */
    void make_complex(ptr_param<complex_type> result) const;

    /**
     * Creates a new real matrix and extracts the real part of the original
     * matrix into that.
     */
    std::unique_ptr<real_type> get_real() const;

    /**
     * Extracts the real part of the original matrix into a given real matrix.
     */
    void get_real(ptr_param<real_type> result) const;

    /**
     * Creates a new real matrix and extracts the imaginary part of the
     * original matrix into that.
     */
    std::unique_ptr<real_type> get_imag() const;

    /**
     * Extracts the imaginary part of the original matrix into a given real
     * matrix.
     */
    void get_imag(ptr_param<real_type> result) const;

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
     * @copydoc Dense::at(size_type, size_type)
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
     * @copydoc Dense::at(size_type)
     */
    ValueType at(size_type idx) const noexcept
    {
        return values_.get_const_data()[linearize_index(idx)];
    }

    /**
     * Scales the matrix with a scalar (aka: BLAS scal).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     */
    void scale(ptr_param<const LinOp> alpha);

    /**
     * Scales the matrix with the inverse of a scalar.
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by 1 / alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the inverse
     *               of the i-th element of alpha (the number of columns of
     *               alpha has to match the number of columns of the matrix).
     */
    void inv_scale(ptr_param<const LinOp> alpha);

    /**
     * Adds `b` scaled by `alpha` to the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, the entire matrix is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of the matrix is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    void add_scaled(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b);

    /**
     * Subtracts `b` scaled by `alpha` from the matrix (aka: BLAS axpy).
     *
     * @param alpha  If alpha is 1x1 Dense matrix, b is scaled
     *               by alpha. If it is a Dense row vector of values,
     *               then i-th column of b is scaled with the i-th
     *               element of alpha (the number of columns of alpha has to
     *               match the number of columns of the matrix).
     * @param b  a matrix of the same dimension as this
     */
    void sub_scaled(ptr_param<const LinOp> alpha, ptr_param<const LinOp> b);

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_dot(ptr_param<const LinOp> b, ptr_param<LinOp> result) const;

    /**
     * Computes the column-wise dot product of this matrix and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_dot(ptr_param<const LinOp> b, ptr_param<LinOp> result,
                     array<char>& tmp) const;

    /**
     * Computes the column-wise dot product of `conj(this matrix)` and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     */
    void compute_conj_dot(ptr_param<const LinOp> b,
                          ptr_param<LinOp> result) const;

    /**
     * Computes the column-wise dot product of `conj(this matrix)` and `b`.
     *
     * @param b  a Dense matrix of same dimension as this
     * @param result  a Dense row vector, used to store the dot product
     *                (the number of column in the vector must match the number
     *                of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_conj_dot(ptr_param<const LinOp> b, ptr_param<LinOp> result,
                          array<char>& tmp) const;

    /**
     * Computes the column-wise Euclidean (L^2) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm2(ptr_param<LinOp> result) const;

    /**
     * Computes the column-wise Euclidean (L^2) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_norm2(ptr_param<LinOp> result, array<char>& tmp) const;

    /**
     * Computes the column-wise (L^1) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_norm1(ptr_param<LinOp> result) const;

    /**
     * Computes the column-wise (L^1) norm of this matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_norm1(ptr_param<LinOp> result, array<char>& tmp) const;

    /**
     * Computes the square of the column-wise Euclidean (L^2) norm of this
     * matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_squared_norm2(ptr_param<LinOp> result) const;

    /**
     * Computes the square of the column-wise Euclidean (L^2) norm of this
     * matrix.
     *
     * @param result  a Dense row vector, used to store the norm
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_squared_norm2(ptr_param<LinOp> result, array<char>& tmp) const;

    /**
     * Computes the column-wise arithmetic mean of this matrix.
     *
     * @param result  a Dense row vector, used to store the mean
     *                (the number of columns in the vector must match the number
     *                of columns of this)
     */
    void compute_mean(ptr_param<LinOp> result) const;

    /**
     * Computes the column-wise arithmetic mean of this matrix.
     *
     * @param result  a Dense row vector, used to store the mean
     *                (the number of columns in the vector must match the
     *                number of columns of this)
     * @param tmp  the temporary storage to use for partial sums during the
     *             reduction computation. It may be resized and/or reset to the
     *             correct executor.
     */
    void compute_mean(ptr_param<LinOp> result, array<char>& tmp) const;

    /**
     * Create a submatrix from the original matrix.
     * Warning: defining stride for this create_submatrix method might cause
     * wrong memory access. Better use the create_submatrix(rows, columns)
     * method instead.
     *
     * @param rows     row span
     * @param columns  column span
     * @param stride   stride of the new submatrix.
     */
    std::unique_ptr<Dense> create_submatrix(const span& rows,
                                            const span& columns,
                                            const size_type stride)
    {
        return this->create_submatrix_impl(rows, columns, stride);
    }

    /**
     * Create a submatrix from the original matrix.
     *
     * @param rows     row span
     * @param columns  column span
     */
    std::unique_ptr<Dense> create_submatrix(const span& rows,
                                            const span& columns)
    {
        return create_submatrix(rows, columns, this->get_stride());
    }

    /**
     * Create a real view of the (potentially) complex original matrix.
     * If the original matrix is real, nothing changes. If the original matrix
     * is complex, the result is created by viewing the complex matrix with as
     * real with a reinterpret_cast with twice the number of columns and
     * double the stride.
     */
    std::unique_ptr<real_type> create_real_view();

    /**
     * @copydoc create_real_view()
     */
    std::unique_ptr<const real_type> create_real_view() const;

    /**
     * Creates a constant (immutable) Dense matrix from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the dimensions of the matrix
     * @param values  the value array of the matrix
     * @param stride  the row-stride of the matrix
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const Dense> create_const(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        gko::detail::const_array_view<ValueType>&& values, size_type stride)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const Dense>(new Dense{
            exec, size, gko::detail::array_const_cast(std::move(values)),
            stride});
    }

    /**
     * Copy-assigns a Dense matrix. Preserves the executor, reallocates the
     * matrix with minimal stride if the dimensions don't match, then copies the
     * data over, ignoring padding.
     */
    Dense& operator=(const Dense&);

    /**
     * Move-assigns a Dense matrix. Preserves the executor, moves the data over
     * preserving size and stride. Leaves the moved-from object in an empty
     * state (0x0 with empty Array).
     */
    Dense& operator=(Dense&&);

    /**
     * Copy-constructs a Dense matrix. Inherits executor and dimensions, but
     * copies data without padding.
     */
    Dense(const Dense&);

    /**
     * Move-constructs a Dense matrix. Inherits executor, dimensions and data
     * with padding. The moved-from object is empty (0x0 with empty Array).
     */
    Dense(Dense&&);

protected:
    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     */
    Dense(std::shared_ptr<const Executor> exec, const dim<2>& size = dim<2>{})
        : Dense(std::move(exec), size, size[1])
    {}

    /**
     * Creates an uninitialized Dense matrix of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the matrix
     * @param stride  stride of the rows (i.e. offset between the first
     *                  elements of two consecutive rows, expressed as the
     *                  number of matrix elements)
     */
    Dense(std::shared_ptr<const Executor> exec, const dim<2>& size,
          size_type stride)
        : EnableLinOp<Dense>(exec, size),
          values_(exec, size[0] * stride),
          stride_(stride)
    {}

    /**
     * Creates a Dense matrix from an already allocated (and initialized) array.
     *
     * @tparam ValuesArray  type of array of values
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
     */
    template <typename ValuesArray>
    Dense(std::shared_ptr<const Executor> exec, const dim<2>& size,
          ValuesArray&& values, size_type stride)
        : EnableLinOp<Dense>(exec, size),
          values_{exec, std::forward<ValuesArray>(values)},
          stride_{stride}
    {
        if (size[0] > 0 && size[1] > 0) {
            GKO_ENSURE_IN_BOUNDS((size[0] - 1) * stride + size[1] - 1,
                                 values_.get_size());
        }
    }

    /**
     * Creates a Dense matrix with the same size and stride as the callers
     * matrix.
     *
     * @returns a Dense matrix with the same size and stride as the caller.
     */
    virtual std::unique_ptr<Dense> create_with_same_config() const
    {
        return Dense::create(this->get_executor(), this->get_size(),
                             this->get_stride());
    }

    /**
     * Creates a Dense matrix with the same type as the callers matrix.
     *
     * @param size  size of the matrix
     *
     * @returns a Dense matrix with the same type as the caller.
     */
    virtual std::unique_ptr<Dense> create_with_type_of_impl(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        size_type stride) const
    {
        return Dense::create(exec, size, stride);
    }

    /**
     * Creates a Dense matrix where the underlying array is a view of this'
     * array.
     *
     * @return  A Dense matrix that is a view of this.
     */
    virtual std::unique_ptr<Dense> create_view_of_impl()
    {
        auto exec = this->get_executor();
        return Dense::create(
            exec, this->get_size(),
            gko::make_array_view(exec, this->get_num_stored_elements(),
                                 this->get_values()),
            this->get_stride());
    }

    /**
     * Creates a immutable Dense matrix where the underlying array is a view of
     * this' array.
     *
     * @return  A immutable Dense matrix that is a view of this.
     */
    virtual std::unique_ptr<const Dense> create_const_view_of_impl() const
    {
        auto exec = this->get_executor();
        return Dense::create_const(
            exec, this->get_size(),
            gko::make_const_array_view(exec, this->get_num_stored_elements(),
                                       this->get_const_values()),
            this->get_stride());
    }

    template <typename IndexType>
    void convert_impl(Coo<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Csr<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Ell<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Fbcsr<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Hybrid<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(Sellp<ValueType, IndexType>* result) const;

    template <typename IndexType>
    void convert_impl(SparsityCsr<ValueType, IndexType>* result) const;

    /**
     * @copydoc scale(const LinOp *)
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void scale_impl(const LinOp* alpha);

    /**
     * @copydoc inv_scale(const LinOp *)
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void inv_scale_impl(const LinOp* alpha);

    /**
     * @copydoc add_scaled(const LinOp *, const LinOp *)
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void add_scaled_impl(const LinOp* alpha, const LinOp* b);

    /**
     * @copydoc sub_scaled(const LinOp *, const LinOp *)
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void sub_scaled_impl(const LinOp* alpha, const LinOp* b);

    /**
     * @copydoc compute_dot(const LinOp*, LinOp*) const
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void compute_dot_impl(const LinOp* b, LinOp* result) const;

    /**
     * @copydoc compute_conj_dot(const LinOp*, LinOp*) const
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void compute_conj_dot_impl(const LinOp* b, LinOp* result) const;

    /**
     * @copydoc compute_norm2(LinOp*) const
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void compute_norm2_impl(LinOp* result) const;

    /**
     * @copydoc compute_norm1(LinOp*) const
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void compute_norm1_impl(LinOp* result) const;

    /**
     * @copydoc compute_squared_norm2(LinOp*) const
     *
     * @deprecated  This function will be removed in the future,
     *              we will instead always use Ginkgo's implementation.
     */
    virtual void compute_squared_norm2_impl(LinOp* result) const;

    /**
     * @copydoc compute_mean(LinOp*) const
     */
    virtual void compute_mean_impl(LinOp* result) const;

    /**
     * Resizes the matrix to the given size.
     *
     * If the new size matches the current size, the stride will be left
     * unchanged, otherwise it will be set to the number of columns.
     *
     * @param new_size  the new matrix dimensions
     */
    void resize(gko::dim<2> new_size);

    /**
     * @copydoc create_submatrix(const span, const span, const size_type)
     *
     * @note  Other implementations of dense should override this function
     *        instead of create_submatrix(const span, const span, const
     *        size_type).
     */
    virtual std::unique_ptr<Dense> create_submatrix_impl(
        const span& rows, const span& columns, const size_type stride);

    void apply_impl(const LinOp* b, LinOp* x) const override;

    void apply_impl(const LinOp* alpha, const LinOp* b, const LinOp* beta,
                    LinOp* x) const override;

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
                      permute_mode mode, Dense* output) const;

    template <typename IndexType>
    void permute_impl(const Permutation<IndexType>* row_permutation,
                      const Permutation<IndexType>* col_permutation,
                      bool invert, Dense* output) const;

    template <typename IndexType>
    void scale_permute_impl(
        const ScaledPermutation<ValueType, IndexType>* permutation,
        permute_mode mode, Dense* output) const;

    template <typename IndexType>
    void scale_permute_impl(
        const ScaledPermutation<ValueType, IndexType>* row_permutation,
        const ScaledPermutation<ValueType, IndexType>* column_permutation,
        bool invert, Dense* output) const;

    template <typename OutputType, typename IndexType>
    void row_gather_impl(const array<IndexType>* row_idxs,
                         Dense<OutputType>* row_collection) const;

    template <typename OutputType, typename IndexType>
    void row_gather_impl(const Dense<ValueType>* alpha,
                         const array<IndexType>* row_idxs,
                         const Dense<ValueType>* beta,
                         Dense<OutputType>* row_collection) const;

private:
    array<value_type> values_;
    size_type stride_;

    void add_scaled_identity_impl(const LinOp* a, const LinOp* b) override;
};


}  // namespace matrix


namespace detail {


template <typename ValueType>
struct temporary_clone_helper<matrix::Dense<ValueType>> {
    static std::unique_ptr<matrix::Dense<ValueType>> create(
        std::shared_ptr<const Executor> exec, matrix::Dense<ValueType>* ptr,
        bool copy_data)
    {
        if (copy_data) {
            return gko::clone(std::move(exec), ptr);
        } else {
            return matrix::Dense<ValueType>::create(exec, ptr->get_size());
        }
    }
};


}  // namespace detail


/**
 * Creates a view of a given Dense vector.
 *
 * @tparam VecPtr  a (smart or raw) pointer to the vector.
 *
 * @param vector  the vector on which to create the view
 */
template <typename VecPtr>
std::unique_ptr<matrix::Dense<typename detail::pointee<VecPtr>::value_type>>
make_dense_view(VecPtr&& vector)
{
    using value_type = typename detail::pointee<VecPtr>::value_type;
    return matrix::Dense<value_type>::create_view_of(vector);
}


/**
 * Creates a view of a given Dense vector.
 *
 * @tparam VecPtr  a (smart or raw) pointer to the vector.
 *
 * @param vector  the vector on which to create the view
 */
template <typename VecPtr>
std::unique_ptr<
    const matrix::Dense<typename detail::pointee<VecPtr>::value_type>>
make_const_dense_view(VecPtr&& vector)
{
    using value_type = typename detail::pointee<VecPtr>::value_type;
    return matrix::Dense<value_type>::create_const_view_of(vector);
}


/**
 * Creates and initializes a column-vector.
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
 * @param vals  values used to initialize the vector
 * @param exec  Executor associated to the vector
 * @param create_args  additional arguments passed to Matrix::create, not
 *                     including the Executor, which is passed as the first
 *                     argument
 *
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride, std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    auto tmp = dense::create(exec->get_master(), dim<2>{num_rows, 1}, stride);
    size_type idx = 0;
    for (const auto& elem : vals) {
        tmp->at(idx) = elem;
        ++idx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}

/**
 * Creates and initializes a column-vector.
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
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<typename Matrix::value_type> vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return initialize<Matrix>(1, vals, std::move(exec),
                              std::forward<TArgs>(create_args)...);
}


/**
 * Creates and initializes a matrix.
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
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    size_type stride,
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    using dense = matrix::Dense<typename Matrix::value_type>;
    size_type num_rows = vals.size();
    size_type num_cols = num_rows > 0 ? begin(vals)->size() : 1;
    auto tmp =
        dense::create(exec->get_master(), dim<2>{num_rows, num_cols}, stride);
    size_type ridx = 0;
    for (const auto& row : vals) {
        size_type cidx = 0;
        for (const auto& elem : row) {
            tmp->at(ridx, cidx) = elem;
            ++cidx;
        }
        ++ridx;
    }
    auto mtx = Matrix::create(exec, std::forward<TArgs>(create_args)...);
    tmp->move_to(mtx);
    return mtx;
}


/**
 * Creates and initializes a matrix.
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
 * @ingroup LinOp
 */
template <typename Matrix, typename... TArgs>
std::unique_ptr<Matrix> initialize(
    std::initializer_list<std::initializer_list<typename Matrix::value_type>>
        vals,
    std::shared_ptr<const Executor> exec, TArgs&&... create_args)
{
    return initialize<Matrix>(vals.size() > 0 ? begin(vals)->size() : 0, vals,
                              std::move(exec),
                              std::forward<TArgs>(create_args)...);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_DENSE_HPP_
