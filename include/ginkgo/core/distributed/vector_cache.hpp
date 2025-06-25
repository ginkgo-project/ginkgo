// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_CACHE_HPP_
#define GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_CACHE_HPP_


#include <memory>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/vector.hpp>


#if GINKGO_BUILD_MPI


namespace gko {
namespace experimental {
namespace distributed {
namespace detail {


/**
 * Manages a distributed vector that is buffered and reused internally to avoid
 * repeated allocations. Copying an instance will only yield an empty object
 * since copying the cached vector would not make sense. The stored object is
 * always mutable, so the cache can be used in a const-context.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
template <typename ValueType>
class VectorCache {
public:
    VectorCache() = default;
    ~VectorCache() = default;
    VectorCache(const VectorCache&) {}
    VectorCache(VectorCache&&) noexcept {}
    VectorCache& operator=(const VectorCache&) { return *this; }
    VectorCache& operator=(VectorCache&&) noexcept { return *this; }
    mutable std::unique_ptr<Vector<ValueType>> vec{};


    /**
     * Initializes the buffered vector with the same configuration as the
     * template vector, if
     * - the current vector is null,
     * - the sizes of the buffered and template vector differ,
     * - the executor of the buffered and template vector differ.
     *
     * @note This does not copy any data from the template vector. If only the
     *       local size differs, only reallocate the local vector not the global
     *       vector.
     *
     * @param template_vec  Defines the configuration (executor, size, stride)
     *                      of the buffered vector.
     */
    void init_from(const Vector<ValueType>* template_vec) const;

    /**
     * Initializes the buffered vector, if
     * - the current vector is null,
     * - the sizes differ,
     * - the executor differs.
     *
     * @param exec  Executor associated with the buffered vector
     * @param comm  Communicator associated with the buffered vector
     * @param global_size  Global size of the buffered vector
     * @param local_size  Processor-local size of the buffered vector, uses
     *                    local_size[1] as the stride
     */
    void init(std::shared_ptr<const Executor> exec,
              gko::experimental::mpi::communicator comm, dim<2> global_size,
              dim<2> local_size) const;

    /**
     * Reference access to the underlying vector.
     *
     * @return  Reference to the stored vector.
     */
    Vector<ValueType>& operator*() const { return *vec; }

    /**
     * Pointer access to the underlying vector.
     * @return  Pointer to the stored vector.
     */
    Vector<ValueType>* operator->() const { return vec.get(); }

    /**
     * Pointer access to the underlying vector.
     * @return  Pointer to the stored vector.
     */
    Vector<ValueType>* get() const { return vec.get(); }
};


// helper to access private member for testing
class GenericVectorCacheAccessor;


/**
 * Manages a distributed vector with different value_type that is buffered and
 * reused internally to avoid repeated allocations. Copying an instance will
 * only yield an empty object since copying the cached vector would not make
 * sense. The stored object is always mutable, so the cache can be used in a
 * const-context.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
class GenericVectorCache {
public:
    friend class GenericVectorCacheAccessor;

    GenericVectorCache() = default;
    ~GenericVectorCache() = default;
    GenericVectorCache(const GenericVectorCache&);
    GenericVectorCache(GenericVectorCache&&) noexcept;
    GenericVectorCache& operator=(const GenericVectorCache&);
    GenericVectorCache& operator=(GenericVectorCache&&) noexcept;

    /**
     * Pointer access to the distributed vector view with specific type on the
     * underlying workspace Initializes the workspace, if
     * - the workspace is null,
     * - the sizes differ,
     * - the executor differs.
     *
     * @param exec  Executor associated with the buffered vector
     * @param comm  Communicator associated with the buffered vector
     * @param global_size  Global size of the buffered vector
     * @param local_size  Processor-local size of the buffered vector, uses
     *                    local_size[1] as the stride
     *
     * @return  Pointer to the vector view.
     */
    template <typename ValueType>
    std::shared_ptr<Vector<ValueType>> get(
        std::shared_ptr<const Executor> exec,
        gko::experimental::mpi::communicator comm, dim<2> global_size,
        dim<2> local_size) const;

private:
    mutable array<char> workspace;
};


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif  // GINKGO_BUILD_MPI
#endif  // GKO_PUBLIC_CORE_DISTRIBUTED_VECTOR_CACHE_HPP_
