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


/** Specifies how a permutation will be applied to a matrix. */
enum class permute_mode {
    /** Neither rows nor columns will be permuted. */
    none = 0b0,
    /** The rows will be permuted. */
    rows = 0b1,
    /** The columns will be permuted. */
    columns = 0b10,
    /**
     * The rows and columns will be permuted. This is equivalent to
     * `permute_mode::rows | permute_mode::columns`.
     */
    symmetric = 0b11,
    /** The permutation will be inverted before being applied. */
    inverse = 0b100,
    /**
     * The rows will be permuted using the inverse permutation. This is
     * equivalent to `permute_mode::rows | permute_mode::inverse`.
     */
    inverse_rows = 0b101,
    /**
     * The columns will be permuted using the inverse permutation. This is
     * equivalent to `permute_mode::columns | permute_mode::inverse`.
     */
    inverse_columns = 0b110,
    /**
     * The rows and columns will be permuted using the inverse permutation. This
     * is equivalent to `permute_mode::symmetric | permute_mode::inverse`.
     */
    inverse_symmetric = 0b111
};


/** Combines two permutation modes. */
inline permute_mode operator|(permute_mode a, permute_mode b)
{
    return static_cast<permute_mode>(static_cast<int>(a) | static_cast<int>(b));
}


/** Computes the intersection of two permutation modes. */
inline permute_mode operator&(permute_mode a, permute_mode b)
{
    return static_cast<permute_mode>(static_cast<int>(a) & static_cast<int>(b));
}


inline std::ostream& operator<<(std::ostream& stream, permute_mode mode)
{
    switch (mode) {
    case permute_mode::none:
        return stream << "none";
    case permute_mode::rows:
        return stream << "rows";
    case permute_mode::columns:
        return stream << "columns";
    case permute_mode::symmetric:
        return stream << "symmetric";
    case permute_mode::inverse:
        return stream << "inverse";
    case permute_mode::inverse_rows:
        return stream << "inverse_rows";
    case permute_mode::inverse_columns:
        return stream << "inverse_columns";
    case permute_mode::inverse_symmetric:
        return stream << "inverse_symmetric";
    }
    return stream;
}


/** @internal std::bitset allows to store any number of bits */
using mask_type = gko::uint64;

static constexpr mask_type row_permute = mask_type{1};
static constexpr mask_type column_permute = mask_type{1 << 2};
static constexpr mask_type inverse_permute = mask_type{1 << 3};

/**
 * Permutation is a matrix "format" which stores the row and column permutation
 * arrays which can be used for re-ordering the rows and columns a matrix.
 *
 * @tparam IndexType  precision of permutation array indices.
 *
 * @note This format is used mainly to allow for an abstraction of the
 * permutation/re-ordering and provides the user with an apply method which
 * calls the respective LinOp's permute operation if the respective LinOp
 * implements the Permutable interface. As such it only stores an array of the
 * permutation indices.
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
    [[deprecated("use get_size()[0] instead")]] size_type get_permutation_size()
        const noexcept;

    [[deprecated("permute mask is no longer supported")]] mask_type
    get_permute_mask() const;

    [[deprecated("permute mask is no longer supported")]] void set_permute_mask(
        mask_type permute_mask);

    /**
     * Returns the inverse permutation.
     *
     * @return a newly created Permutation object storing the inverse
     *         permutation of this Permutation.
     */
    std::unique_ptr<Permutation> invert() const;

    /**
     * Combines this permutation with another permutation via composition.
     * The resulting permutation fulfills `result[i] = other[this[i]]`
     * or `result = other * this` from the matrix perspective, which is
     * equivalent to first permuting by `this` and then by `other`.
     *
     * @param other  the other permutation
     * @return the combined permutation
     */
    std::unique_ptr<Permutation> combine(
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
    [[deprecated(
        "use create_const without size and permute mask")]] static std::
        unique_ptr<const Permutation>
        create_const(std::shared_ptr<const Executor> exec, size_type size,
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
                array<index_type> permutation_indices);

    [[deprecated(
        "dim<2> is no longer supported as a dimension parameter, use size_type "
        "instead")]] Permutation(std::shared_ptr<const Executor> exec,
                                 const dim<2>& size);

    [[deprecated("permute mask is no longer supported")]] Permutation(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        const mask_type& enabled_permute);

    [[deprecated("use the overload without dimensions")]] Permutation(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<index_type> permutation_indices);

    [[deprecated("permute mask is no longer supported")]] Permutation(
        std::shared_ptr<const Executor> exec, const dim<2>& size,
        array<index_type> permutation_indices,
        const mask_type& enabled_permute);

    void apply_impl(const LinOp* in, LinOp* out) const override;

    void apply_impl(const LinOp*, const LinOp* in, const LinOp*,
                    LinOp* out) const override;

private:
    array<index_type> permutation_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_PERMUTATION_HPP_
