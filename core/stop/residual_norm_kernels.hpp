// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_
#define GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace kernels {
namespace residual_norm {


#define GKO_DECLARE_RESIDUAL_NORM_KERNEL(_type)                                \
    void residual_norm(                                                        \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Dense<_type>* tau, const matrix::Dense<_type>* orig_tau, \
        _type rel_residual_goal, uint8 stoppingId, bool setFinalized,          \
        array<stopping_status>* stop_status, array<bool>* device_storage,      \
        bool* all_converged, bool* one_changed)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_RESIDUAL_NORM_KERNEL(ValueType)


}  // namespace residual_norm


namespace implicit_residual_norm {


#define GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL(_type)           \
    void implicit_residual_norm(                                   \
        std::shared_ptr<const DefaultExecutor> exec,               \
        const matrix::Dense<_type>* tau,                           \
        const matrix::Dense<remove_complex<_type>>* orig_tau,      \
        remove_complex<_type> rel_residual_goal, uint8 stoppingId, \
        bool setFinalized, array<stopping_status>* stop_status,    \
        array<bool>* device_storage, bool* all_converged, bool* one_changed)


#define GKO_DECLARE_ALL_AS_TEMPLATES2 \
    template <typename ValueType>     \
    GKO_DECLARE_IMPLICIT_RESIDUAL_NORM_KERNEL(ValueType)


}  // namespace implicit_residual_norm


namespace omp {
namespace residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace residual_norm


namespace implicit_residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES2;

}  // namespace implicit_residual_norm
}  // namespace omp


namespace cuda {
namespace residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace residual_norm


namespace implicit_residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES2;

}  // namespace implicit_residual_norm
}  // namespace cuda


namespace reference {
namespace residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace residual_norm


namespace implicit_residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES2;

}  // namespace implicit_residual_norm
}  // namespace reference


namespace hip {
namespace residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace residual_norm


namespace implicit_residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES2;

}  // namespace implicit_residual_norm
}  // namespace hip


namespace dpcpp {
namespace residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace residual_norm


namespace implicit_residual_norm {

GKO_DECLARE_ALL_AS_TEMPLATES2;

}  // namespace implicit_residual_norm
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES
#undef GKO_DECLARE_ALL_AS_TEMPLATES2

}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_STOP_RESIDUAL_NORM_KERNELS_HPP_
