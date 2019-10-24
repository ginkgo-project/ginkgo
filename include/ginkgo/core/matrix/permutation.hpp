/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_PERMUTATION_HPP_
#define GKO_CORE_MATRIX_PERMUTATION_HPP_


#include <algorithm>
#include <memory>
#include <numeric>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {
namespace matrix {


/**
 * Permutation is a matrix format which stores the row and column permutation
 * arrays which can be used for re-ordering the rows and columns a matrix.
 *
 * @tparam IndexType  precision of permutation array indices.
 *
 * @note This format is used mainly to allow for an abstraction of the
 * permutation/re-ordering and provides the user with an apply method which
 * calls the respective LinOp's permute operation if the respective LinOp
 * implements the Permutable interface.
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
    index_type *get_permutation() noexcept { return permutation_.get_data(); }

    /**
     * @copydoc get_permutation()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_permutation() const noexcept
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

protected:
    /**
     * Creates an uninitialized Permutation arrays on the specified executor..
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
     */
    Permutation(std::shared_ptr<const Executor> exec, const dim<2> &size)
        : EnableLinOp<Permutation>(exec, size),
          permutation_(exec, size[0]),
          row_size_(size[0]),
          col_size_(size[1]),
          permute_inverse_(false),
          permute_row_(true),
          permute_column_(false)
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
     *
     * @note If `indices` is not an rvalue, not an array of IndexType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename IndicesArray>
    Permutation(std::shared_ptr<const Executor> exec, const dim<2> &size,
                IndicesArray &&permutation_indices,
                bool permute_inverse = false, bool permute_row = true,
                bool permute_column = false)
        : EnableLinOp<Permutation>(exec, size),
          permutation_{exec, std::forward<IndicesArray>(permutation_indices)},
          row_size_(size[0]),
          col_size_(size[1]),
          permute_inverse_(permute_inverse),
          permute_row_(permute_row),
          permute_column_(permute_column)
    {
        if (permute_row_) {
            GKO_ENSURE_IN_BOUNDS(size[0] - 1, permutation_.get_num_elems());
        }
        if (permute_column_) {
            GKO_ENSURE_IN_BOUNDS(size[1] - 1, permutation_.get_num_elems());
        }
    }

    void apply_impl(const LinOp *b, LinOp *x) const
    {
        auto perm = as<Permutable<index_type>>(b);
        std::unique_ptr<gko::LinOp> tmp{};
        if (!permute_inverse_) {
            if (permute_row_) {
                tmp = perm->row_permute(&permutation_);
            }
            if (permute_column_) {
                tmp = perm->column_permute(&permutation_);
            }
        } else {
            if (permute_row_) {
                tmp = perm->inverse_row_permute(&permutation_);
            }
            if (permute_column_) {
                tmp = perm->inverse_column_permute(&permutation_);
            }
        }
        x->copy_from(std::move(tmp));
    }


    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const
    {}


private:
    Array<index_type> permutation_;
    size_type row_size_;
    size_type col_size_;
    bool permute_row_;
    bool permute_column_;
    bool permute_inverse_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_PERMUTATION_HPP_
