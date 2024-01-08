// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_
#define GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/permutation.hpp>


namespace gko {
namespace matrix {


/**
 * ScaledPermutation is a matrix combining a permutation with scaling factors.
 * It is a combination of Diagonal and Permutation, and can be read as
 * $SP = P \cdot S$, i.e. the scaling gets applied before the permutation.
 *
 * @tparam IndexType  index type of permutation indices
 * @tparam ValueType  value type of the scaling factors
 *
 * @ingroup permutation
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ScaledPermutation final
    : public EnableLinOp<ScaledPermutation<ValueType, IndexType>>,
      public WritableToMatrixData<ValueType, IndexType> {
    friend class EnablePolymorphicObject<ScaledPermutation, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns a pointer to the scaling factors.
     *
     * @return the pointer to the scaling factors.
     */
    value_type* get_scaling_factors() noexcept { return scale_.get_data(); }

    /**
     * @copydoc get_scaling_factors()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_scaling_factors() const noexcept
    {
        return scale_.get_const_data();
    }

    /**
     * Returns a pointer to the permutation indices.
     *
     * @return the pointer to the permutation indices.
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
     * Returns the inverse of this operator as a scaled permutation.
     * It is computed via $(P S)^-1 = P^{-1} (P S P^{-1})$.
     *
     * @return a newly created ScaledPermutation object storing the inverse
     *         of the permutation and scaling factors of this
     *         ScaledPermutation.
     */
    std::unique_ptr<ScaledPermutation> compute_inverse() const;

    /**
     * Composes this scaled permutation with another scaled permutation. This
     * means `result = other * this` from the matrix perspective, which is
     * equivalent to first scaling and permuting by `this` and then by `other`.
     *
     * @param other  the other permutation
     * @return the combined permutation
     */
    std::unique_ptr<ScaledPermutation> compose(
        ptr_param<const ScaledPermutation> other) const;

    void write(gko::matrix_data<value_type, index_type>& data) const override;

    /**
     * Creates an uninitialized ScaledPermutation matrix.
     *
     * @param exec  Executor associated to the matrix
     * @param size  dimensions of the (square) scaled permutation matrix
     */
    static std::unique_ptr<ScaledPermutation> create(
        std::shared_ptr<const Executor> exec, size_type size = 0);

    /**
     * Create a ScaledPermutation from a Permutation.
     * The permutation will be copied, the scaling factors are all set to 1.0.
     *
     * @param permutation  the permutation
     * @return  the scaled permutation.
     */
    static std::unique_ptr<ScaledPermutation> create(
        ptr_param<const Permutation<IndexType>> permutation);

    /**
     * Creates a ScaledPermutation matrix from already allocated arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param permutation_indices  array of permutation indices
     * @param scaling_factors  array of scaling factors
     */
    static std::unique_ptr<ScaledPermutation> create(
        std::shared_ptr<const Executor> exec, array<value_type> scaling_factors,
        array<index_type> permutation_indices);

    /**
     * Creates a constant (immutable) ScaledPermutation matrix from constant
     * arrays.
     *
     * @param exec  the executor to create the object on
     * @param perm_idxs  the permutation index array of the matrix
     * @param scale  the scaling factor array
     * @returns A smart pointer to the constant matrix wrapping the input arrays
     *          (if it resides on the same executor as the matrix) or a copy of
     *          the arrays on the correct executor.
     */
    static std::unique_ptr<const ScaledPermutation> create_const(
        std::shared_ptr<const Executor> exec,
        gko::detail::const_array_view<value_type>&& scale,
        gko::detail::const_array_view<index_type>&& perm_idxs);

private:
    ScaledPermutation(std::shared_ptr<const Executor> exec, size_type size = 0);

    ScaledPermutation(std::shared_ptr<const Executor> exec,
                      array<value_type> scaling_factors,
                      array<index_type> permutation_indices);

    void apply_impl(const LinOp* in, LinOp* out) const override;

    void apply_impl(const LinOp*, const LinOp* in, const LinOp*,
                    LinOp* out) const override;

    array<value_type> scale_;
    array<index_type> permutation_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_
