// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/batch_lin_op.hpp>


namespace gko::batch_template {

template <typename Concrete>
class EnableBatchUserLinOp : public batch::BatchLinOp {
public:
    using value_type = typename Concrete::value_type;

    void apply(ptr_param<const batch::MultiVector<value_type>> b,
               ptr_param<batch::MultiVector<value_type>> x) const
    {
        this->validate_application_parameters(b.get(), x.get());
        auto exec = self()->get_executor();
        this->template log<log::Logger::batch_solver_completed>(
            log_data_->iter_counts, log_data_->res_norms);
    }

private:
    GKO_ENABLE_SELF(Concrete);
};

}  // namespace gko::batch_template
