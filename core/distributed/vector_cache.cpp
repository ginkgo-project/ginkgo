// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/distributed/vector_cache.hpp"

#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/distributed/vector.hpp>
#include <ginkgo/core/matrix/dense.hpp>

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


#define GKO_DECLARE_VECTOR_CACHE(_type) class VectorCache<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_VECTOR_CACHE);


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko
