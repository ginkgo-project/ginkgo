// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
#define GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_


#include <map>
#include <memory>
#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {


class LinOp;


namespace matrix {


template <typename ValueType>
class Dense;


}


namespace detail {


/**
 * Manages a Dense vector that is buffered and reused internally to avoid
 * repeated allocations. Copying an instance will only yield an empty object
 * since copying the cached vector would not make sense. The stored object is
 * always mutable, so the cache can be used in a const-context.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
template <typename ValueType>
struct DenseCache {
    DenseCache() = default;
    ~DenseCache() = default;
    DenseCache(const DenseCache&) {}
    DenseCache(DenseCache&&) noexcept {}
    DenseCache& operator=(const DenseCache&) { return *this; }
    DenseCache& operator=(DenseCache&&) noexcept { return *this; }
    mutable std::unique_ptr<matrix::Dense<ValueType>> vec{};


    /**
     * Initializes the buffered vector with the same configuration as the
     * template vector, if
     * - the current vector is null,
     * - the sizes of the buffered and template vector differ,
     * - the executor of the buffered and template vector differ.
     *
     * @note This does not copy any data from the template vector.
     *
     * @param template_vec  Defines the configuration (executor, size, stride)
     *                      of the buffered vector.
     */
    void init_from(const matrix::Dense<ValueType>* template_vec) const;

    /**
     * Initializes the buffered vector, if
     * - the current vector is null,
     * - the sizes differ,
     * - the executor differs.
     *
     * @param exec  Executor of the buffered vector.
     * @param size  Size of the buffered vector.
     */
    void init(std::shared_ptr<const Executor> exec, dim<2> size) const;

    /**
     * Reference access to the underlying vector.
     * @return  Reference to the stored vector.
     */
    matrix::Dense<ValueType>& operator*() const { return *vec; }

    /**
     * Pointer access to the underlying vector.
     * @return  Pointer to the stored vector.
     */
    matrix::Dense<ValueType>* operator->() const { return vec.get(); }

    /**
     * Pointer access to the underlying vector.
     * @return  Pointer to the stored vector.
     */
    matrix::Dense<ValueType>* get() const { return vec.get(); }
};


// helper to access private member for testing
class GenericDenseCacheAccessor;


/**
 * Manages a workspace to give Dense Vector with different value_type. The
 * workspace is buffered and reused internally to avoid repeated allocations.
 * Copying an instance will only yield an empty object since copying the cached
 * vector would not make sense. The stored object is always mutable, so the
 * cache can be used in a const-context.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
struct GenericDenseCache {
    friend class GenericDenseCacheAccessor;

    GenericDenseCache() = default;
    ~GenericDenseCache() = default;
    GenericDenseCache(const GenericDenseCache&);
    GenericDenseCache(GenericDenseCache&&) noexcept;
    GenericDenseCache& operator=(const GenericDenseCache&);
    GenericDenseCache& operator=(GenericDenseCache&&) noexcept;

    /**
     * Pointer access to the underlying vector with specific type.
     *
     * @return  Pointer to the vector view.
     */
    template <typename ValueType>
    std::shared_ptr<matrix::Dense<ValueType>> get(
        std::shared_ptr<const Executor> exec, dim<2> size) const;

private:
    mutable array<char> workspace;
};


// helper to access private member for testing.
class ScalarCacheAccessor;


/**
 * Manages a map to store Dense Scalar with different value_type by a
 * user-specified value. The workspace is buffered and reused internally to
 * avoid repeated allocations. Copying an instance will only yield an empty
 * object since copying the cached vector would not make sense. The stored
 * object is always mutable, so the cache can be used in a const-context.
 *
 * @internal  The struct is present to wrap cache-like buffer storage that will
 *            not be copied when the outer object gets copied.
 */
struct ScalarCache {
    friend class ScalarCacheAccessor;

    ScalarCache(std::shared_ptr<const Executor> executor, double scalar_value);
    ~ScalarCache() = default;
    ScalarCache(const ScalarCache& other);
    ScalarCache(ScalarCache&& other) noexcept;
    ScalarCache& operator=(const ScalarCache& other);
    ScalarCache& operator=(ScalarCache&& other) noexcept;

    /**
     * Pointer access to the underlying vector with specific type.
     *
     * @return  Pointer to the vector view.
     */
    template <typename ValueType>
    std::shared_ptr<const matrix::Dense<ValueType>> get() const;

private:
    std::shared_ptr<const Executor> exec;
    double value;
    mutable std::map<std::string, std::shared_ptr<const gko::LinOp>> scalars;
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
