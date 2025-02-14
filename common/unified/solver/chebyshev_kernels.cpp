// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/chebyshev_kernels.hpp"

#include <type_traits>

#include <ginkgo/core/matrix/dense.hpp>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace chebyshev {


template <typename T>
using coeff_type = gko::kernels::chebyshev::coeff_type<T>;


#if GINKGO_DPCPP_SINGLE_MODE


// we only change type in device code to keep the interface is the same as the
// other backend.
template <typename coeff_type>
using if_single_only_type =
    std::conditional_t<std::is_same_v<coeff_type, double>, float,
                       std::complex<float>>;


#else


template <typename T>
struct type_identity {
    using type = T;
};


template <typename coeff_type>
using if_single_only_type = typename type_identity<coeff_type>::type;


#endif


template <typename ValueType>
void init_update(std::shared_ptr<const DefaultExecutor> exec,
                 const coeff_type<ValueType> alpha,
                 const matrix::Dense<ValueType>* inner_sol,
                 matrix::Dense<ValueType>* update_sol,
                 matrix::Dense<ValueType>* output)
{
    using actual_coeff_type = if_single_only_type<coeff_type<ValueType>>;
    using type = device_type<actual_coeff_type>;

    auto alpha_val = static_cast<actual_coeff_type>(alpha);

    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto inner_sol,
                      auto update_sol, auto output) {
            const auto inner_val = static_cast<type>(inner_sol(row, col));
            update_sol(row, col) =
                static_cast<device_type<ValueType>>(inner_val);
            output(row, col) = static_cast<device_type<ValueType>>(
                static_cast<type>(output(row, col)) +
                static_cast<type>(alpha) * inner_val);
        },
        output->get_size(), alpha_val, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL);


template <typename ValueType>
void update(std::shared_ptr<const DefaultExecutor> exec, const ScalarType alpha,
            const ScalarType beta, matrix::Dense<ValueType>* inner_sol,
            matrix::Dense<ValueType>* update_sol,
            matrix::Dense<ValueType>* output)
{
    using actual_coeff_type = if_single_only_type<coeff_type<ValueType>>;
    using type = device_type<actual_coeff_type>;

    auto alpha_val = static_cast<actual_coeff_type>(alpha);
    auto beta_val = static_cast<actual_coeff_type>(beta);

    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto alpha, auto beta, auto inner_sol,
                      auto update_sol, auto output) {
            const auto val = static_cast<type>(inner_sol(row, col)) +
                             static_cast<type>(beta) *
                                 static_cast<type>(update_sol(row, col));
            inner_sol(row, col) = static_cast<device_type<ValueType>>(val);
            update_sol(row, col) = static_cast<device_type<ValueType>>(val);
            output(row, col) = static_cast<device_type<ValueType>>(
                static_cast<type>(output(row, col)) +
                static_cast<type>(alpha) * val);
        },
        output->get_size(), alpha_val, beta_val, inner_sol, update_sol, output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL);


}  // namespace chebyshev
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
