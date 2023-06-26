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

#ifndef GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
#define GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_


#include <any>
#include <memory>


#include <ginkgo/core/base/executor.hpp>


namespace gko {
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
    void init_from(const matrix::Dense<ValueType>* template_vec,
                   gko::array<char>* storage = {}) const;

    /**
     * Initializes the buffered vector, if
     * - the current vector is null,
     * - the sizes differ,
     * - the executor differs.
     *
     * @param exec  Executor of the buffered vector.
     * @param size  Size of the buffered vector.
     */
    void init(std::shared_ptr<const Executor> exec, dim<2> size,
              gko::array<char>* storage = {}) const;

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


struct AnyDenseCache {
    AnyDenseCache() = default;
    ~AnyDenseCache() = default;
    AnyDenseCache(const AnyDenseCache&) {}
    AnyDenseCache(AnyDenseCache&&) noexcept {}
    AnyDenseCache& operator=(const AnyDenseCache&) { return *this; }
    AnyDenseCache& operator=(AnyDenseCache&&) noexcept { return *this; }
    mutable std::any vec{};


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
    void init_from(const matrix::Dense<ValueType>* template_vec) const
    {
        auto concrete = get<ValueType>();
        if (concrete && concrete->get_size() == template_vec->get_size() &&
            concrete->get_executor() == template_vec->get_executor()) {
            return;
        }
        vec = std::move(
            *matrix::Dense<ValueType>::create_with_config_of(template_vec));
    }

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
    void init(std::shared_ptr<const Executor> exec, dim<2> size) const
    {
        auto concrete = get<ValueType>();
        if (concrete && concrete->get_size() == size &&
            concrete->get_executor() == exec) {
            return;
        }
        vec = std::move(*matrix::Dense<ValueType>::create(exec, size));
    }

    /**
     * Pointer access to the underlying vector.
     * @return  Pointer to the stored vector. Returns nullptr if the stored
     * object has a different value type.
     */
    template <typename ValueType>
    matrix::Dense<ValueType>* get() const
    {
        return std::any_cast<matrix::Dense<ValueType>>(&vec);
    }
};

}  // namespace detail
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DENSE_CACHE_HPP_
