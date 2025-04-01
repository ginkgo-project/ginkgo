// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/dense_cache.hpp"

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


GenericDenseCache::GenericDenseCache(const GenericDenseCache&) {}


GenericDenseCache::GenericDenseCache(GenericDenseCache&&) noexcept {}


GenericDenseCache& GenericDenseCache::operator=(const GenericDenseCache&)
{
    return *this;
}


GenericDenseCache& GenericDenseCache::operator=(GenericDenseCache&&) noexcept
{
    return *this;
}


template <typename ValueType>
std::shared_ptr<matrix::Dense<ValueType>> GenericDenseCache::get(
    std::shared_ptr<const Executor> exec, dim<2> size) const
{
    if (exec != workspace.get_executor() ||
        size[0] * size[1] * sizeof(ValueType) > workspace.get_size()) {
        auto new_workspace =
            gko::array<char>(exec, size[0] * size[1] * sizeof(ValueType));
        // We use swap here, otherwise array copy/move between different
        // executor will keep the original executor.
        std::swap(workspace, new_workspace);
    }
    return matrix::Dense<ValueType>::create(
        exec, size,
        make_array_view(exec, size[0] * size[1],
                        reinterpret_cast<ValueType*>(workspace.get_data())),
        size[1]);
}


#define GKO_DECLARE_DENSE_CACHE(_type) struct DenseCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CACHE);

#define GKO_DECLARE_GENERIC_DENSE_CACHE_GET(_type)                       \
    std::shared_ptr<matrix::Dense<_type>> GenericDenseCache::get<_type>( \
        std::shared_ptr<const Executor>, dim<2>) const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GENERIC_DENSE_CACHE_GET);


}  // namespace detail
}  // namespace gko
