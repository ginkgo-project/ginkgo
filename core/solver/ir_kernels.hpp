// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_IR_KERNELS_HPP_
#define GKO_CORE_SOLVER_IR_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace ir {


#define GKO_DECLARE_IR_INITIALIZE_KERNEL                         \
    void initialize(std::shared_ptr<const DefaultExecutor> exec, \
                    array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES GKO_DECLARE_IR_INITIALIZE_KERNEL


}  // namespace ir


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(ir, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_IR_KERNELS_HPP_
