// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_DISTRIBUTED_VECTOR_CACHE_ACCESSOR_HPP_
#define GKO_CORE_DISTRIBUTED_VECTOR_CACHE_ACCESSOR_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/distributed/vector_cache.hpp>


#if GINKGO_BUILD_MPI


namespace gko {
namespace experimental {
namespace distributed {
namespace detail {


// helper to access private member for testing
class GenericVectorCacheAccessor {
public:
    // access to the workspace
    static const array<char>& get_workspace(
        const gko::experimental::distributed::detail::GenericVectorCache&
            cache);
};


}  // namespace detail
}  // namespace distributed
}  // namespace experimental
}  // namespace gko


#endif
#endif  // GKO_CORE_DISTRIBUTED_VECTOR_CACHE_ACCESSOR_HPP_
