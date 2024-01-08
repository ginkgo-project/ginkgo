// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    IndexType get_size() const { return parents_.get_size(); }

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
