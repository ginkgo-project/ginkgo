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

#ifndef GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_
#define GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace matrix {


/**
 * ScaledPermutation is a matrix combining a permutation with scaling factors.
 * It is a combination of Diagonal and Permutation, and can be read as
 * $SP = S \cdot P$, i.e. the scaling gets applied after the permutation.
 *
 * @tparam IndexType  index type of permutation indices
 * @tparam ValueType  value type of the scaling factors
 *
 * @ingroup permutation
 * @ingroup mat_formats
 * @ingroup LinOp
 */
template <typename ValueType = default_precision, typename IndexType = int32>
class ScaledPermutation
    : public EnableLinOp<ScaledPermutation<ValueType, IndexType>>,
      public EnableCreateMethod<ScaledPermutation<ValueType, IndexType>>,
      public WritableToMatrixData<ValueType, IndexType> {
    friend class EnableCreateMethod<ScaledPermutation>;
    friend class EnablePolymorphicObject<ScaledPermutation, LinOp>;

public:
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Returns a pointer to the scaling factors.
     *
     * @return the pointer to the scaling factors.
     */
    value_type* get_scale() noexcept { return scale_.get_data(); }

    /**
     * @copydoc get_scale()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const value_type* get_const_scale() const noexcept
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
     * Returns the inverse scaled permutation.
     *
     * @return a newly created ScaledPermutation object storing the inverse
     *         permutation and scaling factors of this ScalingPermutation.
     */
    std::unique_ptr<ScaledPermutation> invert() const;

    void write(gko::matrix_data<value_type, index_type>& data) const override;

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

protected:
    /**
     * Creates an uninitialized ScaledPermutation matrix.
     *
     * @param exec  Executor associated to the matrix
     * @param size  dimensions of the (square) scaled permutation matrix
     */
    ScaledPermutation(std::shared_ptr<const Executor> exec, size_type size = 0);

    /**
     * Creates a ScaledPermutation matrix from already allocated (and
     * initialized) arrays.
     *
     * @param exec  Executor associated to the matrix
     * @param permutation_indices  array of permutation indices
     * @param scaling_factors  array of scaling factors
     */
    ScaledPermutation(std::shared_ptr<const Executor> exec,
                      array<value_type> scaling_factors,
                      array<index_type> permutation_indices);

    void apply_impl(const LinOp* in, LinOp* out) const override;


    void apply_impl(const LinOp*, const LinOp* in, const LinOp*,
                    LinOp* out) const override;


private:
    array<value_type> scale_;
    array<index_type> permutation_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_SCALED_PERMUTATION_HPP_
