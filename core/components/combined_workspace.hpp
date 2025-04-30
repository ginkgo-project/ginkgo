// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_COMBINED_WORKSPACE_HPP_
#define GKO_CORE_COMPONENTS_COMBINED_WORKSPACE_HPP_


#include <memory>
#include <numeric>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {

// This can be enabled for debugging purposes
#if 1

template <typename IndexType>
struct combined_workspace {
    explicit combined_workspace(std::shared_ptr<const Executor> exec,
                                std::vector<size_type> sizes)
    {
        for (auto size : sizes) {
            arrays.emplace_back(exec, size);
        }
    }

    static size_type get_total_size(std::vector<size_type> sizes)
    {
        return std::accumulate(sizes.begin(), sizes.end(), size_type{});
    }

    IndexType* get_pointer(int i) { return arrays.at(i).get_data(); }

    size_type get_size(int i) const { return arrays.at(i).get_size(); }

    array<IndexType> get_view(int i) { return arrays.at(i).as_view(); }

    std::vector<array<IndexType>> arrays;
};

#else

template <typename IndexType>
struct combined_workspace {
    explicit combined_workspace(std::shared_ptr<const Executor> exec,
                                std::vector<size_type> sizes)
        : offsets{std::move(sizes)}, workspace{exec}
    {
        offsets.insert(offsets.begin(), 0);
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        workspace.resize_and_reset(offsets.back());
    }

    static size_type get_total_size(std::vector<size_type> sizes)
    {
        return std::accumulate(sizes.begin(), sizes.end(), size_type{});
    }

    IndexType* get_pointer(int i)
    {
        return workspace.get_data() + offsets.at(i);
    }
    size_type get_size(int i) const
    {
        return offsets.at(i + 1) - offsets.at(i);
    }
    array<IndexType> get_view(int i)
    {
        return make_array_view(workspace.get_executor(), get_size(i),
                               get_pointer(i));
    }

    array<IndexType> workspace;
    std::vector<size_type> offsets;
};

#endif


}  // namespace gko


#endif  // GKO_CORE_COMPONENTS_COMBINED_WORKSPACE_HPP_
