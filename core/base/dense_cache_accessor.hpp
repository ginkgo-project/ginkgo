// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DENSE_CACHE_ACCESSOR_HPP_
#define GKO_CORE_BASE_DENSE_CACHE_ACCESSOR_HPP_


#include <iostream>
#include <map>
#include <string>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/dense_cache.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/lin_op.hpp>


namespace gko {
namespace detail {


// helper to access private member for testing
class GenericDenseCacheAccessor {
public:
    // access to the workspace
    static const array<char>& get_workspace(const GenericDenseCache& cache);
};


// helper to access private member for testing
class ScalarCacheAccessor {
public:
    // access to the executor
    static std::shared_ptr<const Executor> get_executor(
        const ScalarCache& cache);

    // access to the value
    static double get_value(const ScalarCache& cache);

    // access to the scalars
    static const std::map<std::string, std::shared_ptr<const gko::LinOp>>&
    get_scalars(const ScalarCache& cache);
};


}  // namespace detail
}  // namespace gko


#endif  // GKO_CORE_BASE_DENSE_CACHE_ACCESSOR_HPP_
