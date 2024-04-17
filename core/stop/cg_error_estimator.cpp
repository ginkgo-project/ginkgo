// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/stop/cg_error_estimator.hpp>


#include <numeric>


#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/criterion.hpp>

namespace gko {
namespace stop {


template <typename ValueType>
remove_complex<ValueType> find_safety_factor(
    const std::vector<ValueType>& curve, const std::vector<ValueType>& delta,
    int k)
{
    int ind = 0;
    constexpr remove_complex<ValueType> threshold = 1e-4;
    for (int i = 0; i < curve.size(); i++) {
        if (abs(curve.at(k) / curve.at(i)) <= threshold) {
            ind = i;
        }
    }
    remove_complex<ValueType> safety_factor = 0;
    for (int i = ind; i < curve.size(); i++) {
        safety_factor = max(abs(curve.at(i) / delta.at(i)), safety_factor);
    }
    return safety_factor;
}


template <typename ValueType>
bool CgErrorEstimator<ValueType>::check_impl(
    uint8 stoppingId, bool setFinalized, array<stopping_status>* stop_status,
    bool* one_changed, const Criterion::Updater& updater)
{
    // only handle one rhs now
    assert((updater.cg_prec_vector_matrix_norm_->get_size() == dim<2>{1, 1}));
    bool result = false;
    auto exec = this->get_executor();
    if (!updater.cg_prec_vector_matrix_norm_) {
        return false;
    }
    assert(delta_.size() == updater.num_iterations_);
    auto prec_vector_matrix_norm = exec->copy_val_to_host(
        as<const matrix::Dense<ValueType>>(updater.cg_prec_vector_matrix_norm_)
            ->get_const_values());
    auto sq_implicit_residual = exec->copy_val_to_host(
        as<const matrix::Dense<ValueType>>(updater.implicit_sq_residual_norm_)
            ->get_const_values());
    auto delta_val =
        sq_implicit_residual * sq_implicit_residual / prec_vector_matrix_norm;
    auto tolerance = parameters_.tolerance;
    delta_.push_back(delta_val);
    curve_.push_back(0.0);
    for (auto& e : curve_) {
        e += delta_val;
    }
    if (updater.num_iterations_ > 0) {
        auto safety_factor =
            find_safety_factor(curve_, delta_, updater.num_iterations_);
        auto num = safety_factor * delta_val;
        auto den = std::accumulate(delta_.begin() + k_, delta_.end(),
                                   zero<ValueType>());
        // TODO: maybe it can be replaced by prefix_sum + search?
        while (d_ >= 0 && abs(num / den) <= parameters_.tau) {
            k_++;
            d_--;
            den = std::accumulate(delta_.begin() + k_, delta_.end(),
                                  zero<ValueType>());
        }
        d_++;
        auto tol_upper_bound = sqrt(abs(den) / (1 - parameters_.tau));
        result = tol_upper_bound < parameters_.tolerance;
    }
    if (result) {
        this->set_all_statuses(stoppingId, setFinalized, stop_status);
        *one_changed = true;
    }
    return result;
}

#define GKO_DECLARE_CG_ERROR_ESTIMATOR(_type) class CgErrorEstimator<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_ERROR_ESTIMATOR);


}  // namespace stop
}  // namespace gko
