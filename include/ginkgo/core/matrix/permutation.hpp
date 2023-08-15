// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
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
                    public EnableCreateMethod<Permutation<IndexType>> {
    friend class EnableCreateMethod<Permutation>;
    friend class EnablePolymorphicObject<Permutation, LinOp>;

public:
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
    size_type get_permutation_size() const noexcept
    {
        return permutation_.get_num_elems();
    }

    /**
     * Get the permute masks
     *
     * @return  permute_mask the permute masks
     */
    mask_type get_permute_mask() const { return enabled_permute_; }

    /**
     * Set the permute masks
     *
     * @param permute_mask the permute masks
     */
    void set_permute_mask(mask_type permute_mask)
    {
        enabled_permute_ = permute_mask;
    }

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
        std::shared_ptr<const Executor> exec, size_type size,
        gko::detail::const_array_view<IndexType>&& perm_idxs,
        mask_type enabled_permute = row_permute)
    {
        // cast const-ness away, but return a const object afterwards,
        // so we can ensure that no modifications take place.
        return std::unique_ptr<const Permutation>(new Permutation{
            exec, size, gko::detail::array_const_cast(std::move(perm_idxs)),
            enabled_permute});
    }

protected:
    /**
     * Creates an uninitialized Permutation arrays on the specified executor.
     *
     * @param exec  Executor associated to the LinOp
     */
    Permutation(std::shared_ptr<const Executor> exec)
        : Permutation(std::move(exec), dim<2>{})
    {}

    /**
     * Creates uninitialized Permutation arrays of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the permutable matrix
     * @param enabled_permute  mask for the type of permutation to apply.
     */
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size,
                const mask_type& enabled_permute = row_permute)
        : EnableLinOp<Permutation>(exec, size),
          permutation_(exec, size[0]),
          row_size_(size[0]),
          col_size_(size[1]),
          enabled_permute_(enabled_permute)
    {}

    /**
     * Creates a Permutation matrix from an already allocated (and initialized)
     * row and column permutation arrays.
     *
     * @tparam IndicesArray  type of array of indices
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
    template <typename IndicesArray>
    Permutation(std::shared_ptr<const Executor> exec, const dim<2>& size,
                IndicesArray&& permutation_indices,
                const mask_type& enabled_permute = row_permute)
        : EnableLinOp<Permutation>(exec, size),
          permutation_{exec, std::forward<IndicesArray>(permutation_indices)},
          row_size_(size[0]),
          col_size_(size[1]),
          enabled_permute_(enabled_permute)
    {
        if (enabled_permute_ & row_permute) {
            GKO_ASSERT_EQ(size[0], permutation_.get_num_elems());
        }
        if (enabled_permute_ & column_permute) {
            GKO_ASSERT_EQ(size[1], permutation_.get_num_elems());
        }
    }

    void apply_impl(const LinOp* in, LinOp* out) const
    {
        auto perm = as<Permutable<index_type>>(in);
        std::unique_ptr<gko::LinOp> tmp{};
        if (enabled_permute_ & inverse_permute) {
            if (enabled_permute_ & row_permute) {
                tmp = perm->inverse_row_permute(&permutation_);
            }
            if (enabled_permute_ & column_permute) {
                if (enabled_permute_ & row_permute) {
                    tmp = as<Permutable<index_type>>(tmp.get())
                              ->inverse_column_permute(&permutation_);
                } else {
                    tmp = perm->inverse_column_permute(&permutation_);
                }
            }
        } else {
            if (enabled_permute_ & row_permute) {
                tmp = perm->row_permute(&permutation_);
            }
            if (enabled_permute_ & column_permute) {
                if (enabled_permute_ & row_permute) {
                    tmp = as<Permutable<index_type>>(tmp.get())->column_permute(
                        &permutation_);
                } else {
                    tmp = perm->column_permute(&permutation_);
                }
            }
        }
        out->move_from(tmp);
    }


    void apply_impl(const LinOp*, const LinOp* in, const LinOp*,
                    LinOp* out) const
    {
        // Ignores alpha and beta and just performs a normal permutation as an
        // advanced apply does not really make sense here.
        this->apply_impl(in, out);
    }


private:
    array<index_type> permutation_;
    size_type row_size_;
    size_type col_size_;
    mask_type enabled_permute_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_PERMUTATION_HPP_
