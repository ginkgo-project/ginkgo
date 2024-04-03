// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
#define GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_


#include <memory>


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

/**
 * @copydoc DenseCache
 *
 * @note Compared to DenseCache, this can store vectors of different value type,
 *       similar to std::any. Access to the stored vector is only possible, if
 *       the stored value type is also known. Otherwise, the access will result
 *       in a nullptr.
 */
struct AnyDenseCache {
    AnyDenseCache() = default;
    ~AnyDenseCache() = default;
    AnyDenseCache(const AnyDenseCache&) {}
    AnyDenseCache(AnyDenseCache&&) noexcept {}
    AnyDenseCache& operator=(const AnyDenseCache&) { return *this; }
    AnyDenseCache& operator=(AnyDenseCache&&) noexcept { return *this; }
    mutable std::unique_ptr<LinOp> vec{};


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
    template <typename ValueType>
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
    template <typename ValueType>
    void init(std::shared_ptr<const Executor> exec, dim<2> size) const;

    /**
     * Pointer access to the underlying vector.
     * @tparam ValueType  the desired value type of the vector
     * @return  Pointer to the stored vector. Returns nullptr if the stored
     *          vector has a different value type than ValueType
     */
    template <typename ValueType>
    matrix::Dense<ValueType>* get() const;
};

}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
