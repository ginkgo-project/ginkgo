// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/vector_cache.hpp"

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/distributed/vector_cache_accessor.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace detail {


template <typename ValueType>
void VectorCache<ValueType>::init(std::shared_ptr<const Executor> exec,
                                  gko::experimental::mpi::communicator comm,
                                  dim<2> global_size, dim<2> local_size) const
{
    if (!vec || vec->get_size() != global_size || vec->get_executor() != exec) {
        vec = Vector<ValueType>::create(exec, comm, global_size, local_size);
    } else if (vec->get_local_vector()->get_size() != local_size) {
        // handle locally to eliminate the mpi call
        vec->local_ =
            std::move(gko::matrix::Dense<ValueType>(exec, local_size));
    }
}


template <typename ValueType>
void VectorCache<ValueType>::init_from(
    const Vector<ValueType>* template_vec) const
{
    if (!vec || vec->get_size() != template_vec->get_size() ||
        vec->get_executor() != template_vec->get_executor()) {
        vec = Vector<ValueType>::create_with_config_of(template_vec);
    } else if (vec->get_local_vector()->get_size() !=
               template_vec->get_local_vector()->get_size()) {
        // handle locally to eliminate the mpi call
        vec->local_ = std::move(gko::matrix::Dense<ValueType>(
            template_vec->get_executor(),
            template_vec->get_local_vector()->get_size(),
            template_vec->get_local_vector()->get_stride()));
    }
}


GenericVectorCache::GenericVectorCache(const GenericVectorCache&) {}


GenericVectorCache::GenericVectorCache(GenericVectorCache&&) noexcept {}


GenericVectorCache& GenericVectorCache::operator=(const GenericVectorCache&)
{
    return *this;
}


GenericVectorCache& GenericVectorCache::operator=(GenericVectorCache&&) noexcept
{
    return *this;
}


void GenericVectorCache::init(std::shared_ptr<const Executor> exec,
                              dim<2> global_size, dim<2> local_size) const
{
    exec_ = exec;
    global_size_ = global_size;
    local_size_ = local_size;
}

template <typename ValueType>
std::shared_ptr<Vector<ValueType>> GenericVectorCache::get(
    gko::experimental::mpi::communicator comm) const
{
    auto required_size = local_size_[0] * local_size_[1] * sizeof(ValueType);
    if (exec_ != workspace.get_executor() ||
        required_size > workspace.get_size()) {
        auto new_workspace = gko::array<char>(exec_, required_size);
        // We use swap here, otherwise array copy/move between different
        // executor will keep the original executor.
        std::swap(workspace, new_workspace);
    }
    return Vector<ValueType>::create(
        exec_, comm, global_size_,
        matrix::Dense<ValueType>::create(
            exec_, local_size_,
            make_array_view(exec_, local_size_[0] * local_size_[1],
                            reinterpret_cast<ValueType*>(workspace.get_data())),
            local_size_[1]));
}


const array<char>& GenericVectorCacheAccessor::get_workspace(
    const GenericVectorCache& cache)
{
    return cache.workspace;
}


#define GKO_DECLARE_VECTOR_CACHE(_type) class VectorCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_VECTOR_CACHE);

class GenericVectorCache;

#define GKO_DECLARE_GENERIC_VECTOR_CACHE_GET(_type)         \
    std::shared_ptr<Vector<_type>> GenericVectorCache::get( \
        gko::experimental::mpi::communicator comm) const

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GENERIC_VECTOR_CACHE_GET);


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
