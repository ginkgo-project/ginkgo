// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/log/batch_logger.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace batch {
namespace log {


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
