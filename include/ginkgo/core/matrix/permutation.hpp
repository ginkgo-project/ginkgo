// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_PERMUTATION_HPP_
#define GKO_PUBLIC_CORE_MATRIX_PERMUTATION_HPP_


#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace matrix {


/**
 * Specifies how a permutation will be applied to a matrix.
 * For the effect of the different permutation
 * modes, see the following table.
 *
 * mode              | entry mapping              | matrix representation
 * ------------------|----------------------------|----------------------
 * none              | $A'(i, j) = A(i, j)$       | $A' = A$
 * rows              | $A'(i, j) = A(p[i], j)$    | $A' = P A$
 * columns           | $A'(i, j) = A(i, p[j])$    | $A' = A P^T$
 * inverse_rows      | $A'(p[i], j) = A(i, j)$    | $A' = P^{-1} A$
 * inverse_columns   | $A'(i, p[j]) = A(i, j)$    | $A' = A P^{-T}$
 * symmetric         | $A'(i, j) = A(p[i], p[j])$ | $A' = P A P^T$
 * inverse_symmetric | $A'(p[i], p[j]) = A(i, j)$ | $A' = P^{-1} A P^{-T}$
 */
enum class permute_mode : unsigned {
    /** Neither rows nor columns will be permuted. */
    none = 0b000u,
    /** The rows will be permuted. */
    rows = 0b001u,
    /** The columns will be permuted. */
    columns = 0b010u,
    /**
     * The rows and columns will be permuted. This is equivalent to
     * `permute_mode::rows | permute_mode::columns`.
     */
    symmetric = 0b011u,
    /** The permutation will be inverted before being applied. */
    inverse = 0b100u,
    /**
     * The rows will be permuted using the inverse permutation. This is
     * equivalent to `permute_mode::rows | permute_mode::inverse`.
     */
    inverse_rows = 0b101u,
    /**
     * The columns will be permuted using the inverse permutation. This is
     * equivalent to `permute_mode::columns | permute_mode::inverse`.
     */
    inverse_columns = 0b110u,
    /**
     * The rows and columns will be permuted using the inverse permutation. This
     * is equivalent to `permute_mode::symmetric | permute_mode::inverse`.
     */
    inverse_symmetric = 0b111u
};


/** Combines two permutation modes. */
permute_mode operator|(permute_mode a, permute_mode b);


/** Computes the intersection of two permutation modes. */
permute_mode operator&(permute_mode a, permute_mode b);


/** Computes the symmetric difference of two permutation modes. */
permute_mode operator^(permute_mode a, permute_mode b);


/** Prints a permutation mode. */
std::ostream& operator<<(std::ostream& stream, permute_mode mode);


using mask_type = gko::uint64;

static constexpr mask_type row_permute = mask_type{1};
GKO_DEPRECATED("permute mask is no longer supported")
static constexpr mask_type column_permute = mask_type{1 << 2};
GKO_DEPRECATED("permute mask is no longer supported")
static constexpr mask_type inverse_permute = mask_type{1 << 3};

/**
 * Permutation is a matrix format that represents a permutation matrix,
 * i.e. a matrix where each row and column has exactly one entry.
 * The matrix can only be applied to Dense inputs, where it represents
 * a row permutation: $A' = PA$ means $A'(i, j) = A(p[i], j)$.
 *
 * @tparam IndexType  precision of permutation array indices.
 *
 * @ingroup permutation
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename IndexType = int32>
class Permutation : public EnableLinOp<Permutation<IndexType>>,
                    public EnableCreateMethod<Permutation<IndexType>>,
                    public WritableToMatrixData<default_precision, IndexType> {
    friend class EnableCreateMethod<Permutation>;
    friend class EnablePolymorphicObject<Permutation, LinOp>;

public:
    // value_type is only available to enable the usage of gko::write
    using value_type = default_precision;
    using index_type = IndexType;

    /**
     * Returns a pointer to the array of permutation.
     *
     * @return the pointer to the row permutation array.
     */
    index_type* get_permutation() noexcept { return permutation_.get_data(); }

    /**
     * @copydoc get_permutation()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_permutation() const noexcept
    {
        return permutation_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the permutation
     * array.
     *
     * @return the number of elements explicitly stored in the permutation
     * array.
     */
    GKO_DEPRECATED("use get_size()[0] instead")
    size_type get_permutation_size() const noexcept;

    GKO_DEPRECATED("permute mask is no longer supported")
    mask_type get_permute_mask() const;

    GKO_DEPRECATED("permute mask is no longer supported")
    void set_permute_mask(mask_type permute_mask);

    /**
     * Returns the inverse permutation.
     *
     * @return a newly created Permutation object storing the inverse
     *         permutation of this Permutation.
     */
    std::unique_ptr<Permutation> compute_inverse() const;

    /**
     * Composes this permutation with another permutation.
     * The resulting permutation fulfills `result[i] = this[other[i]]`
     * or `result = other * this` from the matrix perspective, which is
     * equivalent to first permuting by `this` and then by `other`:
     * Combining permutations $P_1$ and $P_2$ with `P = P_1.combine(P_2)`
     * performs the operation permute(A, P) = permute(permute(A, P_1), P_2).
     *
     * @param other  the other permutation
     * @return the combined permutation
     */
    std::unique_ptr<Permutation> compose(
        ptr_param<const Permutation> other) const;

    void write(gko::matrix_data<value_type, index_type>& data) const override;

    /**
     * Creates a constant (immutable) Permutation matrix from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the size of the square matrix
     * @param perm_idxs  the permutation index array of the matrix
     * @param enabled_permute  the mask describing the type of permutation
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    GKO_DEPRECATED("use create_const without size and permute mask")
    static std::unique_ptr<const Permutation> create_const(
        std::shared_ptr<const Executor> exec, size_type size,
        gko::detail::const_array_view<IndexType>&& perm_idxs,
        mask_type enabled_permute = row_permute);
    /**
     * Creates a constant (immutable) Permutation matrix from a constant array.
     *
     * @param exec  the executor to create the matrix on
     * @param size  the size of the square matrix
     * @param perm_idxs  the permutation index array of the matrix
     * @param enabled_permute  the mask describing the type of permutation
     * @returns A smart pointer to the constant matrix wrapping the input array
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the array on the correct executor.
     */
    static std::unique_ptr<const Permutation> create_const(
        std::shared_ptr<const Executor> exec,
        gko::detail::const_array_view<IndexType>&& perm_idxs);

protected:
    /**
     * Creates an uninitialized Permutation arrays on the specified executor.
     *
     * @param exec  Executor associated to the LinOp
     */
    Permutation(std::shared_ptr<const Executor> exec, size_type = 0);

    /**
     * Creates a Permutation matrix from an already allocated (and initialized)
     * row and column permutation arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the permutation array.
     * @param permutation_indices array of permutation array
     * @param enabled_permute  mask for the type of permutation to apply.
     *
     * @note If `permutation_indices` is not an rvalue, not an array of
     * IndexType, or is on the wrong executor, an internal copy will be created,
     * and the original array data will not be used in the matrix.
     */
    Permutation(std::shared_ptr<const Executor> exec,
                array<IndexType> permutation_indices);

    GKO_DEPRECATED(
        "dim<2> is no longer supported as a dimension parameter, use size_type "
        "instead")
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size);

    GKO_DEPRECATED("permute mask is no longer supported")
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size,
                const mask_type& enabled_permute);

    template <typename IndicesArray>
    GKO_DEPRECATED("use the overload without dimensions")
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size,
                IndicesArray&& permutation_indices)
        : Permutation{exec, array<IndexType>{exec, std::forward<IndicesArray>(
                                                       permutation_indices)}}
    {
        GKO_ASSERT_EQ(size[0], permutation_.get_size());
        GKO_ASSERT_IS_SQUARE_MATRIX(size);
    }

    template <typename IndicesArray>
    GKO_DEPRECATED("permute mask is no longer supported")
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size,
                IndicesArray&& permutation_indices,
                const mask_type& enabled_permute)
        : Permutation{std::move(exec),
                      array<IndexType>{exec, std::forward<IndicesArray>(
                                                 permutation_indices)}}
    {
        GKO_ASSERT_EQ(enabled_permute, row_permute);
        GKO_ASSERT_EQ(size[0], permutation_.get_size());
        GKO_ASSERT_IS_SQUARE_MATRIX(size);
    }

    void apply_impl(const LinOp* in, LinOp* out) const override;

    void apply_impl(const LinOp*, const LinOp* in, const LinOp*,
                    LinOp* out) const override;

private:
    array<index_type> permutation_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_PERMUTATION_HPP_
