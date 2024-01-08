// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/ir_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The IR solver namespace.
 *
 * @ingroup ir
 */
namespace ir {


void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < stop_status->get_size(); ++j) {
        stop_status->get_data()[j].reset();
    }
}


}  // namespace ir
}  // namespace reference
}  // namespace kernels
}  // namespace gko
