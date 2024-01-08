// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/common_gmres_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The common GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace common_gmres {
namespace {


template <typename ValueType>
void calculate_sin_and_cos(matrix::Dense<ValueType>* givens_sin,
                           matrix::Dense<ValueType>* givens_cos,
                           matrix::Dense<ValueType>* hessenberg_iter,
                           size_type iter, const size_type rhs)
{
    if (is_zero(hessenberg_iter->at(iter, rhs))) {
        givens_cos->at(iter, rhs) = zero<ValueType>();
        givens_sin->at(iter, rhs) = one<ValueType>();
    } else {
        auto this_hess = hessenberg_iter->at(iter, rhs);
        auto next_hess = hessenberg_iter->at(iter + 1, rhs);
        const auto scale = abs(this_hess) + abs(next_hess);
        const auto hypotenuse =
            scale * sqrt(abs(this_hess / scale) * abs(this_hess / scale) +
                         abs(next_hess / scale) * abs(next_hess / scale));
        givens_cos->at(iter, rhs) = conj(this_hess) / hypotenuse;
        givens_sin->at(iter, rhs) = conj(next_hess) / hypotenuse;
    }
}


template <typename ValueType>
void givens_rotation(matrix::Dense<ValueType>* givens_sin,
                     matrix::Dense<ValueType>* givens_cos,
                     matrix::Dense<ValueType>* hessenberg_iter, size_type iter,
                     const stopping_status* stop_status)
{
    for (size_type i = 0; i < hessenberg_iter->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type j = 0; j < iter; ++j) {
            auto temp = givens_cos->at(j, i) * hessenberg_iter->at(j, i) +
                        givens_sin->at(j, i) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j + 1, i) =
                -conj(givens_sin->at(j, i)) * hessenberg_iter->at(j, i) +
                conj(givens_cos->at(j, i)) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j, i) = temp;
            // temp             =  cos(j)*hessenberg(j) +
            //                     sin(j)*hessenberg(j+1)
            // hessenberg(j+1)  = -conj(sin(j))*hessenberg(j) +
            //                     conj(cos(j))*hessenberg(j+1)
            // hessenberg(j)    =  temp;
        }

        calculate_sin_and_cos(givens_sin, givens_cos, hessenberg_iter, iter, i);

        hessenberg_iter->at(iter, i) =
            givens_cos->at(iter, i) * hessenberg_iter->at(iter, i) +
            givens_sin->at(iter, i) * hessenberg_iter->at(iter + 1, i);
        hessenberg_iter->at(iter + 1, i) = zero<ValueType>();
        // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
        //                      sin(iter)*hessenberg(iter + 1)
        // hessenberg(iter+1) = 0
    }
}


template <typename ValueType>
void calculate_next_residual_norm(
    matrix::Dense<ValueType>* givens_sin, matrix::Dense<ValueType>* givens_cos,
    matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* residual_norm_collection, size_type iter,
    const stopping_status* stop_status)
{
    for (size_type i = 0; i < residual_norm->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        residual_norm_collection->at(iter + 1, i) =
            -conj(givens_sin->at(iter, i)) *
            residual_norm_collection->at(iter, i);
        residual_norm_collection->at(iter, i) =
            givens_cos->at(iter, i) * residual_norm_collection->at(iter, i);
        residual_norm->at(0, i) =
            abs(residual_norm_collection->at(iter + 1, i));
    }
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                stopping_status* stop_status)
{
    const auto krylov_dim = givens_sin->get_size()[0];
    using NormValueType = remove_complex<ValueType>;
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            residual->at(i, j) = b->at(i, j);
        }
        for (size_type i = 0; i < krylov_dim; ++i) {
            givens_sin->at(i, j) = zero<ValueType>();
            givens_cos->at(i, j) = zero<ValueType>();
        }
        stop_status[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL);


template <typename ValueType>
void hessenberg_qr(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::Dense<ValueType>* givens_sin,
                   matrix::Dense<ValueType>* givens_cos,
                   matrix::Dense<remove_complex<ValueType>>* residual_norm,
                   matrix::Dense<ValueType>* residual_norm_collection,
                   matrix::Dense<ValueType>* hessenberg_iter, size_type iter,
                   size_type* final_iter_nums,
                   const stopping_status* stop_status)
{
    for (size_type i = 0; i < givens_sin->get_size()[1]; ++i) {
        if (!stop_status[i].has_stopped()) {
            final_iter_nums[i]++;
        }
    }

    givens_rotation(givens_sin, givens_cos, hessenberg_iter, iter, stop_status);
    calculate_next_residual_norm(givens_sin, givens_cos, residual_norm,
                                 residual_norm_collection, iter, stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL);


template <typename ValueType>
void solve_krylov(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y, const size_type* final_iter_nums,
                  const stopping_status* stop_status)
{
    for (size_type k = 0; k < residual_norm_collection->get_size()[1]; ++k) {
        if (stop_status[k].is_finalized()) {
            continue;
        }
        for (int i = final_iter_nums[k] - 1; i >= 0; --i) {
            auto temp = residual_norm_collection->at(i, k);
            for (size_type j = i + 1; j < final_iter_nums[k]; ++j) {
                temp -=
                    hessenberg->at(
                        i, j * residual_norm_collection->get_size()[1] + k) *
                    y->at(j, k);
            }
            y->at(i, k) =
                temp / hessenberg->at(
                           i, i * residual_norm_collection->get_size()[1] + k);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace common_gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
