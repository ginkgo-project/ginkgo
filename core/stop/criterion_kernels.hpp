// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_STOP_CRITERION_KERNELS_HPP_
#define GKO_CORE_STOP_CRITERION_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace set_all_statuses {


#define GKO_DECLARE_SET_ALL_STATUSES_KERNEL                            \
    void set_all_statuses(std::shared_ptr<const DefaultExecutor> exec, \
                          uint8 stoppingId, bool setFinalized,         \
                          array<stopping_status>* stop_status)


}  // namespace set_all_statuses


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(set_all_statuses,
                                        GKO_DECLARE_SET_ALL_STATUSES_KERNEL);
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_STOP_CRITERION_KERNELS_HPP_
