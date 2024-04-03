// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/dense_cache.hpp>


#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {


template <typename ValueType>
void DenseCache<ValueType>::init(std::shared_ptr<const Executor> exec,
                                 dim<2> size) const
{
    if (!vec || vec->get_size() != size || vec->get_executor() != exec) {
        vec = matrix::Dense<ValueType>::create(exec, size);
    }
}


template <typename ValueType>
void DenseCache<ValueType>::init_from(
    const matrix::Dense<ValueType>* template_vec) const
{
    if (!vec || vec->get_size() != template_vec->get_size() ||
        vec->get_executor() != template_vec->get_executor()) {
        vec = matrix::Dense<ValueType>::create_with_config_of(template_vec);
    }
}


#define GKO_DECLARE_DENSE_CACHE(_type) class DenseCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CACHE);


template <typename ValueType>
void AnyDenseCache::init_from(
    const matrix::Dense<ValueType>* template_vec) const
{
    if (!vec || vec->get_size() != template_vec->get_size() ||
        vec->get_executor() != template_vec->get_executor() ||
        !this->get<ValueType>()) {
        vec = matrix::Dense<ValueType>::create_with_config_of(template_vec);
    }
}

#define GKO_DECLARE_ANYDENSECACHE_INIT_FROM(_vtype)                          \
    void AnyDenseCache::init_from(const matrix::Dense<_vtype>* template_vec) \
        const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ANYDENSECACHE_INIT_FROM);

#undef GKO_DECLARE_ANYDENSECACHE_INIT_FROM


template <typename ValueType>
void AnyDenseCache::init(std::shared_ptr<const Executor> exec,
                         dim<2> size) const
{
    if (!vec || vec->get_size() != size || vec->get_executor() != exec ||
        !this->get<ValueType>()) {
        vec = matrix::Dense<ValueType>::create(exec, size);
    }
}

#define GKO_DECLARE_ANYDENSECACHE_INIT(_vtype)                             \
    void AnyDenseCache::init<_vtype>(std::shared_ptr<const Executor> exec, \
                                     dim<2> size) const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ANYDENSECACHE_INIT);

#undef GKO_DECLARE_ANYDENSECACHE_INIT


template <typename ValueType>
matrix::Dense<ValueType>* AnyDenseCache::get() const
{
    if (auto p = dynamic_cast<matrix::Dense<ValueType>*>(vec.get())) {
        return p;
    }

    return nullptr;
}

#define GKO_DECLARE_ANYDENSECACHE_GET(_vtype) \
    matrix::Dense<_vtype>* AnyDenseCache::get() const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ANYDENSECACHE_GET);

#undef GKO_DECLARE_ANYDENSECACHE_GET


}  // namespace detail
}  // namespace gko
