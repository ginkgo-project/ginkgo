// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_COMPONENTS_DISJOINT_SETS_HPP_
#define GKO_COMMON_CUDA_HIP_COMPONENTS_DISJOINT_SETS_HPP_

#include <thrust/swap.h>

#include "common/cuda_hip/components/memory.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * Disjoint-set data structure for use on GPUs.
 * It has mainly two sets of functions:
 * 1. functions that can be used safely only in read-only contexts, i.e. when no
 *    thread is accessing non-const member functions.
 * 2. functions that can be used safely in concurrent write contexts, i.e.
 *    when different threads are applying path compression or calling join(...)
 * When joining two sets, the data structure will always attach the smaller
 * element to the larger element, i.e. a set's representative will always be its
 * maximum.
 * When considering find(...) function calls in a concurrent kernel, their
 * result will not always point to the globally agreed-upon representative of a
 * set, to we refer to their results as approximate representatives.
 * This uses some techniques mentioned in J. Jaiganesh and M. Burtscher,
 * "A High-Performance Connected Components Implementation for GPUs."
 * Proceedings of the 2018 ACM International Symposium on High-Performance
 * Parallel and Distributed Computing. June 2018
 */
template <typename IndexType>
class disjoint_sets {
public:
    using index_type = std::remove_const_t<IndexType>;

    __device__ __forceinline__ disjoint_sets(IndexType* parents, IndexType size)
        : parents_{parents}, size_{size}
    {}

    /**
     * Joins the sets containing x and y. Does nothing (except for waste time)
     * if the two sets are already joined. This is the only operation that
     * strictly requires L2 atomics for global correctness. For best
     * performance, x and y should already be (approximate) representatives of
     * their sets. Returns the (approximate) representative of the joined set.
     */
    __device__ __forceinline__ IndexType join(IndexType x, IndexType y)
    {
        check_index(x);
        check_index(y);
        auto new_rep = max(x, y);
        auto old_rep = min(x, y);
        // try to attach old_rep directly to new_rep
        auto old_parent =
            atomic_cas_relaxed(parents_ + old_rep, old_rep, new_rep);
        // if this fails, the parent of old_rep changed recently
        // (or old_rep was only an approximate rep), so we need to try
        // again by updating the parent's parent (hopefully its rep)
        while (old_parent != old_rep && old_rep != new_rep) {
            old_rep = old_parent;
            // ensure that new_rep > old_rep
            if (old_rep > new_rep) {
                thrust::swap(old_rep, new_rep);
            }
            old_parent =
                atomic_cas_relaxed(parents_ + old_rep, old_rep, new_rep);
        }
        return new_rep;
    }

    /**
     * Return the representative of the set containing the given element.
     * This function is only safe in read-only contexts.
     */
    __device__ __forceinline__ IndexType find_weak(IndexType x) const
    {
        check_index(x);
        auto cur = x;
        auto parent = parents_[cur];
        while (parent != cur) {
            cur = parent;
            parent = parents_[cur];
        };
        return parent;
    }

    /**
     * Returns an approximation to the representative of the set containing the
     * given element. This can be used safely in concurrent write contexts, but
     * may return an element that is not actually the representative from the
     * view of a different thread, while still belonging to the same set,
     * because the set's representative can be modified while this loop is
     * running, or an old value can still be cached in L1 cache.
     * It can be used to retrieve better representatives for the join function.
     */
    __device__ __forceinline__ IndexType find_relaxed(IndexType x) const
    {
        check_index(x);
        auto cur = x;
        // here we use L1 atomics because it is cheaper, and we don't need an
        // exact global representative
        auto parent = load_relaxed_local(parents_ + cur);
        while (parent != cur) {
            cur = parent;
            parent = load_relaxed_local(parents_ + cur);
        };
        return parent;
    }

    /**
     * Returns an approximation to the representative of the set containing the
     * given element, while shortening paths to the rep via pointer doubling.
     * This can be used safely in concurrent write contexts, but may return an
     * element that is not actually the representative from the view of a
     * different thread, while still belonging to the same set, because the
     * set's representative can be modified while this loop is running, or an
     * old value can still be cached in L1 cache. It can be used to retrieve
     * better representatives for the join function.
     * TODO fix use when no joins are happening
     */
    __device__ __forceinline__ IndexType
    find_relaxed_compressing(IndexType x) const
    {
        check_index(x);
        auto cur = x;
        // here we use L1 atomics because it is cheaper, and we don't need an
        // exact global representative.
        // During concurrent modifications, this may actually undo the
        // compressions by other SMs due to writing back the L1 cache
        // contents.
        auto parent = load_relaxed_local(parents_ + cur);
        if (cur != parent) {
            auto grandparent = load_relaxed_local(parents_ + parent);
            while (grandparent != parent) {
                // pointer doubling
                // node --> parent --> grandparent
                // turns into
                // node -------------> grandparent
                //                       |
                //          parent ------/
                // This operation is safe, because only the representative of
                // each set will be changed in subsequent operations, and this
                // only shortens paths along intermediate nodes
                store_relaxed_local(parents_ + cur, grandparent);
                cur = parent;
                parent = grandparent;
                grandparent = load_relaxed_local(parents_ + parent);
            }
        }
        return parent;
    }

    /**
     * Compresses the path from x to its representative to a single edge.
     * This is safe to use in a concurrent write context, but will only produce
     * the intended effect if there are no concurrent join operations happening.
     * Otherwise it may not fully compress the path.
     */
    __device__ __forceinline__ void path_compress_relaxed(IndexType x)
    {
        check_index(x);
        parents_[x] = find_relaxed(x);
    }

    /**
     * Returns whether a given element is its set's representative.
     * This function is only safe in read-only contexts.
     */
    __device__ __forceinline__ bool is_representative_weak(IndexType x) const
    {
        check_index(x);
        return parents_[x] == x;
    }

    __device__ __forceinline__ IndexType size() const { return size_; }

private:
    __device__ __forceinline__ void check_index(IndexType i) const
    {
        assert(i >= 0);
        assert(i < size());
    }

    IndexType* parents_;
    IndexType size_;
};


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_COMMON_CUDA_HIP_COMPONENTS_DISJOINT_SETS_HPP_
