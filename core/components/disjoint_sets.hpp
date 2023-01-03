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

#ifndef GKO_CORE_COMPONENTS_DISJOINT_SETS_HPP_
#define GKO_CORE_COMPONENTS_DISJOINT_SETS_HPP_


#include <ginkgo/core/base/array.hpp>


namespace gko {


/**
 * A disjoint-set/union-find data structure.
 * It uses a parent array with path compression and union-by-size to provide
 * almost constant-time join and find operations.
 *
 * @tparam IndexType  the type to store element indices.
 */
template <typename IndexType>
class disjoint_sets {
public:
    /**
     * Constructs a new disjoint sets data structure where all elements are
     * singleton sets.
     *
     * @param size  the number of elements.
     * @param exec  the executor whose associated host executor will be used to
     *              allocate storage.
     */
    explicit disjoint_sets(std::shared_ptr<const Executor> exec, IndexType size)
        : parents_{exec->get_master(), static_cast<size_type>(size)}
    {
        parents_.fill(-1);
    }

    /**
     * Returns true if and only if the element x is the representative of its
     * set.
     *
     * @param x  the element
     * @return true if and only if x is the representative of its set, i.e.
     *         find(x) == x.
     */
    bool is_representative(IndexType x) const { return parent(x) < 0; }

    /**
     * Returns the representative of the set containing element x.
     *
     * @param x  the element
     * @return the element representing the set containing x
     */
    IndexType const_find(IndexType x) const
    {
        while (!is_representative(x)) {
            x = parent(x);
        }
        return x;
    }

    /**
     * Returns the representative of the set containing element x.
     * Also performs path-compression on the corresponding path.
     *
     * @param x  the element
     * @return the element representing the set containing x
     */
    IndexType find(IndexType x)
    {
        auto rep = const_find(x);
        // path compression
        while (!is_representative(x)) {
            auto tmp = parent(x);
            parent(x) = rep;
            x = tmp;
        }
        return rep;
    }

    /**
     * Returns the length of the path from x to its set's representative.
     * This is mostly used for testing purposes, to check whether the union
     * heuristics are working correctly.
     *
     * @param x  the element.
     * @return  the length of the path from x to its set's representative. That
     *          means if x is its own representative, the result is 0.
     */
    IndexType get_path_length(IndexType x) const
    {
        IndexType len{};
        while (!is_representative(x)) {
            x = parent(x);
            len++;
        }
        return len;
    }

    /**
     * Returns the size of the set containing x.
     * This is mostly used for testing purposes,
     * to check whether set sizes are updated correctly.
     *
     * @param x  the element.
     * @return  the number of elements in the set containing x.
     */
    IndexType get_set_size(IndexType x) const { return -parent(const_find(x)); }

    /**
     * Joins the sets containing the given elements if they are disjoint.
     * The representative of the larger set will be used for the resulting set,
     * with the first element being chosen in case of a tie (union-by-size).
     *
     * @param a  the first element
     * @param b  the second element
     * @return the representative of the resulting joint set.
     */
    IndexType join(IndexType a, IndexType b)
    {
        auto pa = find(a);
        auto pb = find(b);
        if (pa != pb) {
            auto sa = -parent(pa);
            auto sb = -parent(pb);
            if (sa < sb) {
                std::swap(pa, pb);
            }
            parent(pb) = pa;
            parent(pa) = -sa - sb;
        }
        return pa;
    }

    /**
     * Returns the number of elements represented in this data structure.
     *
     * @return the number of elements represented in this data structure.
     */
    IndexType get_size() const { return parents_.get_num_elems(); }

private:
    const IndexType& parent(IndexType x) const
    {
        return parents_.get_const_data()[x];
    }

    IndexType& parent(IndexType x) { return parents_.get_data()[x]; }

    array<IndexType> parents_;
};


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_DISJOINT_SETS_HPP_
