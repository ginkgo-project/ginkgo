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
     * Returns a pointer to the array of row permutation.
     *
     * @return the pointer to the row permutation array.
     */
    index_type *get_row_permutation() noexcept
    {
        return row_permutation_.get_data();
    }

    /**
     * Returns a pointer to the array of column permutation.
     *
     * @return the pointer to the column permutation array.
     */
    index_type *get_col_permutation() noexcept
    {
        return col_permutation_.get_data();
    }

    /**
     * @copydoc get_row_permutation()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_row_permutation() const noexcept
    {
        return row_permutation_.get_const_data();
    }

    /**
     * @copydoc get_col_permutation()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type *get_const_col_permutation() const noexcept
    {
        return col_permutation_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the row permutation
     * array.
     *
     * @return the number of elements explicitly stored in the row permutation
     * array.
     */
    size_type get_size_row_permutation() const noexcept
    {
        return row_permutation_.get_num_elems();
    }

    /**
     * Returns the number of elements explicitly stored in the column
     * permutation array.
     *
     * @return the number of elements explicitly stored in the column
     * permutation array.
     */
    size_type get_size_col_permutation() const noexcept
    {
        return col_permutation_.get_num_elems();
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
          row_permutation_(exec, size[0]),
          col_permutation_(exec, size[1]),
          row_size_(size[0]),
          col_size_(size[1])
    {}

    /**
     * Creates a Permutation array from an already allocated (and initialized)
     * row permutation indices array. By default, the column permutation is
     * initialized so that even if the apply is called the columns are not
     * permuted.
     *
     * @tparam IndicesArray  type of array of indices
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the permutation array.
     * @param row_indices array of row permutation array
     *
     * @note If `indices` is not an rvalue, not an array of IndexType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename IndicesArray>
    Permutation(std::shared_ptr<const Executor> exec, const dim<2> &size,
                IndicesArray &&row_indices)
        : EnableLinOp<Permutation>(exec, size),
          row_permutation_{exec, std::forward<IndicesArray>(row_indices)},
          col_permutation_{exec, size[1]},
          row_size_(size[0]),
          col_size_(size[1])
    {
        std::vector<IndexType> t(col_size_, 0);
        std::iota(t.begin(), t.end(), 0);
        col_permutation_ = gko::Array<IndexType>(exec, t.begin(), t.end());
        GKO_ENSURE_IN_BOUNDS(size[0] - 1, row_permutation_.get_num_elems());
        GKO_ENSURE_IN_BOUNDS(size[1] - 1, col_permutation_.get_num_elems());
    }

    /**
     * Creates a Permutation matrix from an already allocated (and initialized)
     * row and column permutation arrays.
     *
     * @tparam IndicesArray  type of array of indices
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the permutation array.
     * @param row_indices array of row permutation array
     * @param col_indices array of col permutation array
     *
     * @note If `indices` is not an rvalue, not an array of IndexType, or is on
     *       the wrong executor, an internal copy will be created, and the
     *       original array data will not be used in the matrix.
     */
    template <typename IndicesArray>
    Permutation(std::shared_ptr<const Executor> exec, const dim<2> &size,
                IndicesArray &&row_indices, IndicesArray &&col_indices)
        : EnableLinOp<Permutation>(exec, size),
          row_permutation_{exec, std::forward<IndicesArray>(row_indices)},
          col_permutation_{exec, std::forward<IndicesArray>(col_indices)},
          row_size_(size[0]),
          col_size_(size[1])
    {
        GKO_ENSURE_IN_BOUNDS(size[0] - 1, row_permutation_.get_num_elems());
        GKO_ENSURE_IN_BOUNDS(size[1] - 1, col_permutation_.get_num_elems());
    }

    void apply_impl(const LinOp *b, LinOp *x) const {}


    void apply_impl(const LinOp *alpha, const LinOp *b, const LinOp *beta,
                    LinOp *x) const
    {}


private:
    Array<index_type> row_permutation_;
    Array<index_type> col_permutation_;
    size_type row_size_;
    size_type col_size_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_PERMUTATION_HPP_
