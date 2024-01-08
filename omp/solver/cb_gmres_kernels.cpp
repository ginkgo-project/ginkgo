// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cb_gmres_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/cb_gmres.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/solver/cb_gmres_accessor.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The CB_GMRES solver namespace.
 *
 * @ingroup cb_gmres
 */
namespace cb_gmres {


namespace {


template <typename ValueType, typename Accessor3d>
void finish_arnoldi_CGS(std::shared_ptr<const OmpExecutor> exec,
                        matrix::Dense<ValueType>* next_krylov_basis,
                        Accessor3d krylov_bases,
                        matrix::Dense<ValueType>* hessenberg_iter,
                        matrix::Dense<ValueType>* buffer_iter,
                        matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
                        size_type iter, const stopping_status* stop_status)
{
    using rc_vtype = remove_complex<ValueType>;
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;
    const rc_vtype eta = 1.0 / sqrt(2.0);

    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }

        auto nrm = zero<rc_vtype>();
        run_kernel_reduction(
            exec,
            [](auto row, auto col, auto next_krylov_basis) {
                return squared_norm(next_krylov_basis(row, col));
            },
            GKO_KERNEL_REDUCE_SUM(rc_vtype), &nrm,
            next_krylov_basis->get_size()[0], static_cast<int64>(i),
            next_krylov_basis);
        arnoldi_norm->at(0, i) = eta * sqrt(nrm);
        // nrmP = norm(next_krylov_basis)
#pragma omp parallel for
        for (size_type k = 0; k < iter + 1; ++k) {
            ValueType hessenberg_iter_entry = zero<ValueType>();
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                hessenberg_iter_entry +=
                    next_krylov_basis->at(j, i) * conj(krylov_bases(k, j, i));
            }
            hessenberg_iter->at(k, i) = hessenberg_iter_entry;
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        for (size_type k = 0; k < iter + 1; ++k) {
#pragma omp parallel for
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                next_krylov_basis->at(j, i) -=
                    hessenberg_iter->at(k, i) * krylov_bases(k, j, i);
            }
        }
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        nrm = zero<rc_vtype>();
        auto inf = zero<rc_vtype>();
        auto result_pair = std::make_pair(nrm, inf);
        run_kernel_reduction(
            exec,
            [](auto row, auto col, auto next_krylov_basis) {
                const auto val = next_krylov_basis(row, col);
                return std::make_pair(squared_norm(val), abs(val));
            },
            [](auto a, auto b) {
                return std::make_pair(a.first + b.first,
                                      std::max(a.second, b.second));
            },
            [](auto a) { return a; }, std::make_pair(rc_vtype{}, rc_vtype{}),
            &result_pair, next_krylov_basis->get_size()[0],
            static_cast<int64>(i), next_krylov_basis);
        nrm = result_pair.first;
        inf = result_pair.second;
        arnoldi_norm->at(1, i) = sqrt(nrm);
        if (has_scalar) {
            arnoldi_norm->at(2, i) = inf;
        }

        for (size_type l = 1;
             (arnoldi_norm->at(1, i)) < (arnoldi_norm->at(0, i)) && l < 3;
             l++) {
            arnoldi_norm->at(0, i) = eta * arnoldi_norm->at(1, i);
            // nrmP = nrmN
#pragma omp parallel for
            for (size_type k = 0; k < iter + 1; ++k) {
                ValueType hessenberg_iter_entry = zero<ValueType>();
                for (size_type j = 0; j < next_krylov_basis->get_size()[0];
                     ++j) {
                    hessenberg_iter_entry += next_krylov_basis->at(j, i) *
                                             conj(krylov_bases(k, j, i));
                }
                buffer_iter->at(k, i) = hessenberg_iter_entry;
            }
            // for i in 1:iter
            //     buffer(iter, i) = next_krylov_basis' * krylov_bases(:, i)
            // end
            for (size_type k = 0; k < iter + 1; ++k) {
#pragma omp parallel for
                for (size_type j = 0; j < next_krylov_basis->get_size()[0];
                     ++j) {
                    next_krylov_basis->at(j, i) -=
                        buffer_iter->at(k, i) * conj(krylov_bases(k, j, i));
                }
            }
            // for i in 1:iter
            //     next_krylov_basis  -= buffer(iter, i) * krylov_bases(:, i)
            // end
            nrm = zero<rc_vtype>();
            inf = zero<rc_vtype>();
            run_kernel_reduction(
                exec,
                [](auto row, auto col, auto next_krylov_basis) {
                    const auto val = next_krylov_basis(row, col);
                    return std::make_pair(squared_norm(val), abs(val));
                },
                [](auto a, auto b) {
                    return std::make_pair(a.first + b.first,
                                          std::max(a.second, b.second));
                },
                [](auto a) { return a; },
                std::make_pair(rc_vtype{}, rc_vtype{}), &result_pair,
                next_krylov_basis->get_size()[0], static_cast<int64>(i),
                next_krylov_basis);
            nrm = result_pair.first;
            inf = result_pair.second;
            arnoldi_norm->at(1, i) = sqrt(nrm);
            if (has_scalar) {
                arnoldi_norm->at(2, i) = inf;
            }
            // nrmN = norm(next_krylov_basis)
            // nrmI = infnorm(next_krylov_basis)
        }
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, iter + 1, i,
            arnoldi_norm->at(2, i) / arnoldi_norm->at(1, i));
        // reorthogonalization
        hessenberg_iter->at(iter + 1, i) = (arnoldi_norm->at(1, i));
        // hessenberg(iter, iter + 1) = norm(next_krylov_basis)
#pragma omp parallel for
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            next_krylov_basis->at(j, i) /= hessenberg_iter->at(iter + 1, i);
            krylov_bases(iter + 1, j, i) = next_krylov_basis->at(j, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // krylov_bases(:, iter + 1) = next_krylov_basis
        // End of arnoldi
    }
}


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
#pragma omp parallel for
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
#pragma omp parallel for
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


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType>* residual_norm_collection,
    const matrix::Dense<ValueType>* hessenberg, matrix::Dense<ValueType>* y,
    const size_type* final_iter_nums)
{
#pragma omp parallel for
    for (size_type k = 0; k < residual_norm_collection->get_size()[1]; ++k) {
        for (int64 i = final_iter_nums[k] - 1; i >= 0; --i) {
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


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(ConstAccessor3d krylov_bases,
                  const matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const size_type* final_iter_nums)
{
#pragma omp parallel for
    for (size_type i = 0; i < before_preconditioner->get_size()[0]; ++i) {
        for (size_type k = 0; k < before_preconditioner->get_size()[1]; ++k) {
            before_preconditioner->at(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner->at(i, k) +=
                    krylov_bases(j, i, k) * y->at(j, k);
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const OmpExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                array<stopping_status>* stop_status, size_type krylov_dim)
{
    using rc_vtype = remove_complex<ValueType>;

    for (size_type j = 0; j < b->get_size()[1]; ++j) {
#pragma omp parallel for
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            residual->at(i, j) = b->at(i, j);
        }

#pragma omp parallel for
        for (size_type i = 0; i < krylov_dim; ++i) {
            givens_sin->at(i, j) = zero<ValueType>();
            givens_cos->at(i, j) = zero<ValueType>();
        }
        stop_status->get_data()[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL);


template <typename ValueType, typename Accessor3d>
void restart(std::shared_ptr<const OmpExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             Accessor3d krylov_bases,
             matrix::Dense<ValueType>* next_krylov_basis,
             array<size_type>* final_iter_nums, array<char>&,
             size_type krylov_dim)
{
    using rc_vtype = remove_complex<ValueType>;
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;

    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        // Calculate residual norm
        auto res_norm = zero<rc_vtype>();
        auto res_inf = zero<rc_vtype>();

        auto result_pair = std::make_pair(res_norm, res_inf);
        run_kernel_reduction(
            exec,
            [](auto row, auto col, auto residual) {
                const auto val = residual(row, col);
                return std::make_pair(squared_norm(val), abs(val));
            },
            [](auto a, auto b) {
                return std::make_pair(a.first + b.first,
                                      std::max(a.second, b.second));
            },
            [](auto a) { return a; }, std::make_pair(rc_vtype{}, rc_vtype{}),
            &result_pair, next_krylov_basis->get_size()[0],
            static_cast<int64>(j), residual);
        res_norm = result_pair.first;
        res_inf = result_pair.second;
        residual_norm->at(0, j) = sqrt(res_norm);
        if (has_scalar) {
            arnoldi_norm->at(2, j) = res_inf;
        }
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, {0}, j,
            arnoldi_norm->at(2, j) / residual_norm->at(0, j));

#pragma omp parallel for
        for (size_type i = 0; i < krylov_dim + 1; ++i) {
            if (i == 0) {
                residual_norm_collection->at(i, j) = residual_norm->at(0, j);
            } else {
                residual_norm_collection->at(i, j) = zero<ValueType>();
            }
        }

#pragma omp parallel for
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            auto value = residual->at(i, j) / residual_norm->at(0, j);
            krylov_bases(0, i, j) = value;
            next_krylov_basis->at(i, j) = value;
        }
        final_iter_nums->get_data()[j] = 0;
    }

#pragma omp parallel for
    for (size_type k = 1; k < krylov_dim + 1; ++k) {
        for (size_type j = 0; j < residual->get_size()[1]; ++j) {
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, k, j, one<rc_vtype>());
        }
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            for (size_type j = 0; j < residual->get_size()[1]; ++j) {
                krylov_bases(k, i, j) = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_RESTART_KERNEL);


template <typename ValueType, typename Accessor3d>
void arnoldi(std::shared_ptr<const OmpExecutor> exec,
             matrix::Dense<ValueType>* next_krylov_basis,
             matrix::Dense<ValueType>* givens_sin,
             matrix::Dense<ValueType>* givens_cos,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             Accessor3d krylov_bases, matrix::Dense<ValueType>* hessenberg_iter,
             matrix::Dense<ValueType>* buffer_iter,
             matrix::Dense<remove_complex<ValueType>>* arnoldi_norm,
             size_type iter, array<size_type>* final_iter_nums,
             const array<stopping_status>* stop_status, array<stopping_status>*,
             array<size_type>*)
{
#pragma omp parallel for
    for (size_type i = 0; i < final_iter_nums->get_size(); ++i) {
        final_iter_nums->get_data()[i] +=
            (1 - static_cast<size_type>(
                     stop_status->get_const_data()[i].has_stopped()));
    }
    finish_arnoldi_CGS(exec, next_krylov_basis, krylov_bases, hessenberg_iter,
                       buffer_iter, arnoldi_norm, iter,
                       stop_status->get_const_data());
    givens_rotation(givens_sin, givens_cos, hessenberg_iter, iter,
                    stop_status->get_const_data());
    calculate_next_residual_norm(givens_sin, givens_cos, residual_norm,
                                 residual_norm_collection, iter,
                                 stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_ARNOLDI_KERNEL);


template <typename ValueType, typename ConstAccessor3d>
void solve_krylov(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  ConstAccessor3d krylov_bases,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums->get_const_data());
    calculate_qy(krylov_bases, y, before_preconditioner,
                 final_iter_nums->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace cb_gmres
}  // namespace omp
}  // namespace kernels
}  // namespace gko
