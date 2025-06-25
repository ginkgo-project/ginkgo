// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/dense_cache.hpp"

#include <memory>
#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dim.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dense_cache_accessor.hpp"

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


const array<char>& GenericDenseCacheAccessor::get_workspace(
    const GenericDenseCache& cache)
{
    return cache.workspace;
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


std::shared_ptr<const Executor> ScalarCacheAccessor::get_executor(
    const ScalarCache& cache)
{
    return cache.exec;
}


double ScalarCacheAccessor::get_value(const ScalarCache& cache)
{
    return cache.value;
}


const std::map<std::string, std::shared_ptr<const gko::LinOp>>&
ScalarCacheAccessor::get_scalars(const ScalarCache& cache)
{
    return cache.scalars;
}


ScalarCache::ScalarCache(std::shared_ptr<const Executor> executor,
                         double scalar_value)
    : exec(std::move(executor)), value(scalar_value){};

ScalarCache::ScalarCache(const ScalarCache& other) { *this = other; }


ScalarCache::ScalarCache(ScalarCache&& other) noexcept
{
    *this = std::move(other);
}


ScalarCache& ScalarCache::operator=(const ScalarCache& other)
{
    exec = other.exec;
    value = other.value;
    return *this;
}


ScalarCache& ScalarCache::operator=(ScalarCache&& other) noexcept
{
    exec = std::exchange(other.exec, nullptr);
    value = std::exchange(other.value, 0.0);
    other.scalars.clear();
    return *this;
}


template <typename ValueType>
std::shared_ptr<const matrix::Dense<ValueType>> ScalarCache::get() const
{
    // using typeid name as key
    std::string value_string = typeid(ValueType).name();
    auto search = scalars.find(value_string);
    if (search != scalars.end()) {
        return std::dynamic_pointer_cast<const matrix::Dense<ValueType>>(
            search->second);
    } else {
        auto new_scalar =
            share(matrix::Dense<ValueType>::create(exec, dim<2>{1, 1}));
        new_scalar->fill(static_cast<ValueType>(value));
        scalars[value_string] = new_scalar;
        return new_scalar;
    }
}


#define GKO_DECLARE_DENSE_CACHE(_type) struct DenseCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CACHE);

#define GKO_DECLARE_GENERIC_DENSE_CACHE_GET(_type)                       \
    std::shared_ptr<matrix::Dense<_type>> GenericDenseCache::get<_type>( \
        std::shared_ptr<const Executor>, dim<2>) const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GENERIC_DENSE_CACHE_GET);

#define GKO_DECLARE_SCALAR_CACHE_GET(_type) \
    std::shared_ptr<const matrix::Dense<_type>> ScalarCache::get<_type>() const
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SCALAR_CACHE_GET);


}  // namespace detail
}  // namespace gko
