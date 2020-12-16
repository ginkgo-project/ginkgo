/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
#define GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_


#include <algorithm>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/**
 * An index set class represents an ordered set object in a mathematical sense.
 * The index set contains subsets which store the starting and end points of a
 * range, [a,b), storing the first index and one past the last index. As the
 * index set only stores the end-points of ranges, it can be quite efficient in
 * terms of storage.
 *
 * This class automatically merges index ranges that are continuous when new
 * indices are added. To access the subsets, two iterators are provided, an
 * Element iterator which can iterate through the elements in the subsets stored
 * in the index set. An interval iterator is also provided to iterate between
 * the the different subsets themselves.
 *
 * @tparam index_type  type of the indices being stored in the index set.
 *
 * @ingroup IndexSet
 */
template <typename IndexType = int32>
class IndexSet {
public:
    using index_type = IndexType;

    IndexSet()
        : index_space_size_(0),
          exec_(nullptr),
          subsets_begin_(exec_),
          subsets_end_(exec_),
          superset_cumulative_indices_(exec_)
    {}

    IndexSet(std::shared_ptr<const gko::Executor> executor,
             const index_type size)
        : index_space_size_(size),
          exec_(executor),
          subsets_begin_(exec_),
          subsets_end_(exec_),
          superset_cumulative_indices_(exec_)
    {}


    IndexSet(std::shared_ptr<const gko::Executor> executor,
             const index_type size, const gko::Array<index_type> &indices)
        : index_space_size_(size),
          exec_(executor),
          subsets_begin_(exec_),
          subsets_end_(exec_),
          superset_cumulative_indices_(exec_)
    {
        this->populate_subsets(indices);
    }

    /**
     * Returns the Executor associated with the index set.
     *
     * @return the Executor associated with the index set
     */
    std::shared_ptr<const gko::Executor> get_executor() const noexcept
    {
        return exec_;
    }

    index_type get_size() const { return this->index_space_size_; }

    bool is_contiguous() const { return (this->get_num_subsets() <= 1); }

    index_type get_num_elems() const { return this->num_stored_indices_; };

    index_type get_global_index(const index_type &local_index) const;

    index_type get_local_index(const index_type &global_index) const;

    Array<index_type> get_global_indices_from_local(
        const Array<index_type> &local_indices) const;

    Array<index_type> get_local_indices_from_global(
        const Array<index_type> &global_indices) const;

    index_type get_num_subsets() const
    {
        return this->subsets_begin_.get_num_elems();
    }

    const index_type *get_subsets_begin() const
    {
        return this->subsets_begin_.get_const_data();
    }

    const index_type *get_subsets_end() const
    {
        return this->subsets_end_.get_const_data();
    }

    const index_type *get_superset_indices() const
    {
        return this->superset_cumulative_indices_.get_const_data();
    }

private:
    void populate_subsets(const gko::Array<index_type> &indices);

    std::shared_ptr<const gko::Executor> exec_;

    index_type index_space_size_;
    index_type num_stored_indices_;
    gko::Array<index_type> subsets_begin_;
    gko::Array<index_type> subsets_end_;
    gko::Array<index_type> superset_cumulative_indices_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_INDEX_SET_HPP_
