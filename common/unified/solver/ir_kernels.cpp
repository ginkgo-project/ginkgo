// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/ir_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The IR solver namespace.
 *
 * @ingroup ir
 */
namespace ir {


void initialize(std::shared_ptr<const DefaultExecutor> exec,
                array<stopping_status>* stop_status)
{
    run_kernel(
        exec, [] GKO_KERNEL(auto i, auto stop) { stop[i].reset(); },
        stop_status->get_size(), *stop_status);
}


}  // namespace ir
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
