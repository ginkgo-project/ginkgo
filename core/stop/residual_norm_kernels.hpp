// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_
#define GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace residual_norm {


#define GKO_DECLARE_RESIDUAL_NORM_KERNEL(ValueType)                            \
    void residual_norm(std::shared_ptr<const DefaultExecutor> exec,            \
                       matrix::view::dense<const ValueType> tau,               \
                       matrix::view::dense<const ValueType> orig_tau,          \
                       ValueType rel_residual_goal, uint8 stoppingId,          \
                       bool setFinalized, array<stopping_status>& stop_status, \
                       array<bool>& device_storage, bool* all_converged,       \
                       bool* one_changed)


#define GKO_DECLARE_ALL_AS_TEMPLATES(_export_macro) \
    template <typename ValueType>                   \
    _export_macro GKO_DECLARE_RESIDUAL_NORM_KERNEL(ValueType)


}  // namespace residual_norm


namespace implicit_residual_norm {


#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL(ValueType)           \
    void implicit_residual_norm(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                   \
        matrix::view::dense<const ValueType> tau,                      \
        matrix::view::dense<const remove_complex<ValueType>> orig_tau, \
        remove_complex<ValueType> rel_residual_goal, uint8 stoppingId, \
        bool setFinalized, array<stopping_status>& stop_status,        \
        array<bool>& device_storage, bool* all_converged, bool* one_changed)


#define GKO_DECLARE_ALL_AS_TEMPLATES2(_export_macro) \
    template <typename ValueType>                    \
    _export_macro GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL(ValueType)


}  // namespace implicit_residual_norm


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(residual_norm,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);

GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(implicit_residual_norm,
                                        GKO_DECLARE_ALL_AS_TEMPLATES2);


#undef GKO_DECLARE_ALL_AS_TEMPLATES
#undef GKO_DECLARE_ALL_AS_TEMPLATES2

}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_
