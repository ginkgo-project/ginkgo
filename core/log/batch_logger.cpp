// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/log/batch_logger.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>

#include "core/base/workspace_aliases.hpp"


namespace gko {
namespace batch {
namespace log {
namespace detail {


template <typename ValueType>
log_data<ValueType>::log_data(std::shared_ptr<const Executor> exec,
                              size_type num_batch_items)
    : res_norms(exec), iter_counts(exec)
{
    if (num_batch_items > 0) {
        iter_counts.resize_and_reset(num_batch_items);
        res_norms.resize_and_reset(num_batch_items);
    } else {
        GKO_INVALID_STATE("Invalid num batch items passed in");
    }
}


template <typename ValueType>
log_data<ValueType>::log_data(std::shared_ptr<const Executor> exec,
                              size_type num_batch_items,
                              array<unsigned char>& workspace)
    : res_norms(exec), iter_counts(exec)
{
    // it should at least `num * (sizeof(real_type) + sizeof(int))` with some
    // additional buffer for alias purpose, but we simply request a large enough
    // size here.
    const size_type reqd_workspace_size = num_batch_items * 32;

    if (num_batch_items > 0 && !workspace.is_owning() &&
        workspace.get_size() >= reqd_workspace_size) {
        gko::detail::layout<2, 8> workspace_alias;
        auto slot_1 = workspace_alias.get_slot(0);
        auto slot_2 = workspace_alias.get_slot(1);
        auto iter_alias = slot_1->create_alias<index_type>(num_batch_items);
        auto res_alias = slot_2->create_alias<real_type>(num_batch_items);

        // Temporary storage mapping
        auto err = workspace_alias.map_to_buffer(workspace.get_data(),
                                                 reqd_workspace_size);
        GKO_ASSERT(err == GKO_DEVICE_NO_ERROR);
        iter_counts =
            array<index_type>::view(exec, num_batch_items, iter_alias.get());
        res_norms =
            array<real_type>::view(exec, num_batch_items, res_alias.get());
    } else {
        GKO_INVALID_STATE("invalid workspace or num batch items passed in");
    }
}

#define GKO_DECLARE_LOG_DATA(_type) struct log_data<_type>

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(GKO_DECLARE_LOG_DATA);

#undef GKO_DECLARE_LOG_DATA


}  // namespace detail


template <typename ValueType>
void BatchConvergence<ValueType>::on_batch_solver_completed(
    const array<int>& iteration_count,
    const array<remove_complex<ValueType>>& residual_norm) const
{
    if (this->iteration_count_.get_size() == 0) {
        this->iteration_count_ = gko::array<int>(iteration_count.get_executor(),
                                                 iteration_count.get_size());
    }
    if (this->residual_norm_.get_size() == 0) {
        this->residual_norm_ = gko::array<remove_complex<ValueType>>(
            residual_norm.get_executor(), residual_norm.get_size());
    }
    this->iteration_count_ = iteration_count;
    this->residual_norm_ = residual_norm;
}


#define GKO_DECLARE_BATCH_CONVERGENCE(_type) class BatchConvergence<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_CONVERGENCE);


}  // namespace log
}  // namespace batch
}  // namespace gko
