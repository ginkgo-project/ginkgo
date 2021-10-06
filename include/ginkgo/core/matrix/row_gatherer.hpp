/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_MATRIX_ROW_GATHERER_HPP_
#define GKO_PUBLIC_CORE_MATRIX_ROW_GATHERER_HPP_


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
 * RowGatherer is a matrix "format" which stores the row rowgatherer
 * arrays which can be used to perform rowgatherer rows to another matrix.
 *
 * @tparam IndexType  precision of rowgatherer array indices.
 *
 * @note This format is used mainly to allow for an abstraction of the
 * rowgatherer and provides the user with an apply method which
 * calls the respective LinOp's rowgatherer operation if the respective LinOp
 * implements the RowGatherable interface. As such it only stores an array of
 * the rowgatherer indices.
 *
 * @ingroup rowgatherer
 * @ingroup matrix
 * @ingroup LinOp
 */
template <typename IndexType = int32>
class RowGatherer : public EnableLinOp<RowGatherer<IndexType>>,
                    public EnableCreateMethod<RowGatherer<IndexType>> {
    friend class EnableCreateMethod<RowGatherer>;
    friend class EnablePolymorphicObject<RowGatherer, LinOp>;

public:
    using index_type = IndexType;

    /**
     * Returns a pointer to the array of row_gather_index.
     *
     * @return the pointer to the row_gather_index array.
     */
    index_type* get_row_gather_index() noexcept
    {
        return row_gather_index_.get_data();
    }

    /**
     * @copydoc get_row_gather_index()
     *
     * @note This is the constant version of the function, which can be
     *       significantly more memory efficient than the non-constant version,
     *       so always prefer this version.
     */
    const index_type* get_const_row_gather_index() const noexcept
    {
        return row_gather_index_.get_const_data();
    }

    /**
     * Returns the number of elements explicitly stored in the row_gather_index
     * array.
     *
     * @return the number of elements explicitly stored in the row_gather_index
     * array.
     */
    size_type get_row_gather_index_size() const noexcept
    {
        return row_gather_index_.get_num_elems();
    }

protected:
    /**
     * Creates an uninitialized RowGatherer arrays on the specified executor..
     *
     * @param exec  Executor associated to the LinOp
     */
    RowGatherer(std::shared_ptr<const Executor> exec)
        : RowGatherer(std::move(exec), dim<2>{})
    {}

    /**
     * Creates uninitialized RowGatherer arrays of the specified size.
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the RowGatherable matrix
     */
    RowGatherer(std::shared_ptr<const Executor> exec, const dim<2>& size)
        : EnableLinOp<RowGatherer>(exec, size), row_gather_index_(exec, size[0])
    {}

    /**
     * Creates a RowGatherer matrix from an already allocated (and initialized)
     * row and column rowgatherer arrays.
     *
     * @tparam IndicesArray  type of array of indices
     *
     * @param exec  Executor associated to the matrix
     * @param size  size of the rowgatherer array.
     * @param row_gather_indexindices array of rowgatherer array
     *
     * @note If `row_gather_index` is not an rvalue, not an array of
     * IndexType, or is on the wrong executor, an internal copy will be created,
     * and the original array data will not be used in the matrix.
     */
    template <typename IndicesArray>
    RowGatherer(std::shared_ptr<const Executor> exec, const dim<2>& size,
                IndicesArray&& row_gather_index)
        : EnableLinOp<RowGatherer>(exec, size),
          row_gather_index_{exec, std::forward<IndicesArray>(row_gather_index)}
    {
        GKO_ASSERT_EQ(size[0], row_gather_index_.get_num_elems());
    }

    void apply_impl(const LinOp* in, LinOp* out) const
    {
        auto gather = gko::as<RowGatherable<index_type>>(in);
        gather->row_gather(&row_gather_index_, out);
    }


    void apply_impl(const LinOp* alpha, const LinOp* in, const LinOp* beta,
                    LinOp* out) const
    {
        auto gather = gko::as<RowGatherable<index_type>>(in);
        // gather->row_gather(alpha, &row_gather_index_, beta, out);
    }

private:
    gko::Array<index_type> row_gather_index_;
};


}  // namespace matrix
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_MATRIX_ROW_GATHERER_HPP_
