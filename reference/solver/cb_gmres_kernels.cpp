// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cb_gmres_kernels.hpp"

#include <type_traits>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/solver/cb_gmres_accessor.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The CB_GMRES solver namespace.
 *
 * @ingroup cb_gmres
 */
namespace cb_gmres {


namespace {


template <typename ValueType, typename Accessor3d>
void finish_arnoldi_CGS(
    matrix::view::dense<ValueType> next_krylov_basis, Accessor3d krylov_bases,
    matrix::view::dense<ValueType> hessenberg_iter,
    matrix::view::dense<ValueType> buffer_iter,
    matrix::view::dense<remove_complex<ValueType>> arnoldi_norm, size_type iter,
    const stopping_status* stop_status)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;
    using rc_vtype = remove_complex<ValueType>;
    const rc_vtype eta = 1.0 / sqrt(2.0);

    for (size_type i = 0; i < next_krylov_basis.size[1]; ++i) {
        arnoldi_norm(0, i) = zero<rc_vtype>();
        for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
            arnoldi_norm(0, i) += squared_norm(next_krylov_basis(j, i));
        }
        arnoldi_norm(0, i) = eta * sqrt(arnoldi_norm(0, i));
        // arnoldi_norm(0, i) = norm(next_krylov_basis)
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type k = 0; k < iter + 1; ++k) {
            hessenberg_iter(k, i) = zero<ValueType>();
            for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
                hessenberg_iter(k, i) +=
                    next_krylov_basis(j, i) * conj(krylov_bases(k, j, i));
            }
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        // end
        for (size_type k = 0; k < iter + 1; ++k) {
            for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
                next_krylov_basis(j, i) -=
                    hessenberg_iter(k, i) * krylov_bases(k, j, i);
            }
        }
        // for i in 1:iter
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end
        arnoldi_norm(1, i) = zero<rc_vtype>();
        if (has_scalar) {
            arnoldi_norm(2, i) = zero<rc_vtype>();
        }
        for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
            arnoldi_norm(1, i) += squared_norm(next_krylov_basis(j, i));
            if (has_scalar) {
                arnoldi_norm(2, i) =
                    (arnoldi_norm(2, i) >= abs(next_krylov_basis(j, i)))
                        ? arnoldi_norm(2, i)
                        : abs(next_krylov_basis(j, i));
            }
        }
        arnoldi_norm(1, i) = sqrt(arnoldi_norm(1, i));

        for (size_type l = 1;
             (arnoldi_norm(1, i)) < (arnoldi_norm(0, i)) && l < 3; l++) {
            arnoldi_norm(0, i) = eta * arnoldi_norm(1, i);
            for (size_type k = 0; k < iter + 1; ++k) {
                buffer_iter(k, i) = zero<ValueType>();
                for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
                    buffer_iter(k, i) +=
                        next_krylov_basis(j, i) * conj(krylov_bases(k, j, i));
                }
            }
            // for i in 1:iter
            //     buffer(iter, i) = next_krylov_basis' * krylov_bases(:, i)
            // end
            for (size_type k = 0; k < iter + 1; ++k) {
                for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
                    next_krylov_basis(j, i) -=
                        buffer_iter(k, i) * conj(krylov_bases(k, j, i));
                }
                hessenberg_iter(k, i) += buffer_iter(k, i);
            }
            // for i in 1:iter
            //     next_krylov_basis   -= buffer(iter, i) * krylov_bases(:, i)
            //     hessenberg(iter, i) += buffer(iter, i)
            // end
            arnoldi_norm(1, i) = zero<rc_vtype>();
            arnoldi_norm(2, i) = zero<rc_vtype>();
            for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
                arnoldi_norm(1, i) += squared_norm(next_krylov_basis(j, i));
                arnoldi_norm(2, i) =
                    (arnoldi_norm(2, i) >= abs(next_krylov_basis(j, i)))
                        ? arnoldi_norm(2, i)
                        : abs(next_krylov_basis(j, i));
            }
            arnoldi_norm(1, i) = sqrt(arnoldi_norm(1, i));
            // nrmN = norm(next_krylov_basis)
        }
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, iter + 1, i, arnoldi_norm(2, i) / arnoldi_norm(1, i));
        hessenberg_iter(iter + 1, i) = arnoldi_norm(1, i);
        // hessenberg(iter, iter + 1) = norm(next_krylov_basis)
        for (size_type j = 0; j < next_krylov_basis.size[0]; ++j) {
            next_krylov_basis(j, i) /= hessenberg_iter(iter + 1, i);
            krylov_bases(iter + 1, j, i) = next_krylov_basis(j, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // krylov_bases(:, iter + 1) = next_krylov_basis
        // End of arnoldi
    }
}


template <typename ValueType>
void calculate_sin_and_cos(matrix::view::dense<ValueType> givens_sin,
                           matrix::view::dense<ValueType> givens_cos,
                           matrix::view::dense<ValueType> hessenberg_iter,
                           size_type iter, const size_type rhs)
{
    if (is_zero(hessenberg_iter(iter, rhs))) {
        givens_cos(iter, rhs) = zero<ValueType>();
        givens_sin(iter, rhs) = one<ValueType>();
    } else {
        auto this_hess = hessenberg_iter(iter, rhs);
        auto next_hess = hessenberg_iter(iter + 1, rhs);
        const auto scale = abs(this_hess) + abs(next_hess);
        const auto hypotenuse =
            scale * sqrt(abs(this_hess / scale) * abs(this_hess / scale) +
                         abs(next_hess / scale) * abs(next_hess / scale));
        givens_cos(iter, rhs) = conj(this_hess) / hypotenuse;
        givens_sin(iter, rhs) = conj(next_hess) / hypotenuse;
    }
}


template <typename ValueType>
void givens_rotation(matrix::view::dense<ValueType> givens_sin,
                     matrix::view::dense<ValueType> givens_cos,
                     matrix::view::dense<ValueType> hessenberg_iter,
                     size_type iter, const stopping_status* stop_status)
{
    for (size_type i = 0; i < hessenberg_iter.size[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type j = 0; j < iter; ++j) {
            auto temp = givens_cos(j, i) * hessenberg_iter(j, i) +
                        givens_sin(j, i) * hessenberg_iter(j + 1, i);
            hessenberg_iter(j + 1, i) =
                -conj(givens_sin(j, i)) * hessenberg_iter(j, i) +
                conj(givens_cos(j, i)) * hessenberg_iter(j + 1, i);
            hessenberg_iter(j, i) = temp;
            // temp             =  cos(j)*hessenberg(j) +
            //                     sin(j)*hessenberg(j+1)
            // hessenberg(j+1)  = -conj(sin(j))*hessenberg(j) +
            //                     conj(cos(j))*hessenberg(j+1)
            // hessenberg(j)    =  temp;
        }

        calculate_sin_and_cos(givens_sin, givens_cos, hessenberg_iter, iter, i);

        hessenberg_iter(iter, i) =
            givens_cos(iter, i) * hessenberg_iter(iter, i) +
            givens_sin(iter, i) * hessenberg_iter(iter + 1, i);
        hessenberg_iter(iter + 1, i) = zero<ValueType>();
        // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
        //                      sin(iter)*hessenberg(iter + 1)
        // hessenberg(iter+1) = 0
    }
}


template <typename ValueType>
void calculate_next_residual_norm(
    matrix::view::dense<ValueType> givens_sin,
    matrix::view::dense<ValueType> givens_cos,
    matrix::view::dense<remove_complex<ValueType>> residual_norm,
    matrix::view::dense<ValueType> residual_norm_collection, size_type iter,
    const stopping_status* stop_status)
{
    for (size_type i = 0; i < residual_norm.size[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        residual_norm_collection(iter + 1, i) =
            -conj(givens_sin(iter, i)) * residual_norm_collection(iter, i);
        residual_norm_collection(iter, i) =
            givens_cos(iter, i) * residual_norm_collection(iter, i);
        residual_norm(0, i) = abs(residual_norm_collection(iter + 1, i));
    }
}


template <typename ValueType>
void solve_upper_triangular(
    matrix::view::dense<const ValueType> residual_norm_collection,
    matrix::view::dense<const ValueType> hessenberg,
    matrix::view::dense<ValueType> y, const size_type* final_iter_nums)
{
    for (size_type k = 0; k < residual_norm_collection.size[1]; ++k) {
        for (int64 i = final_iter_nums[k] - 1; i >= 0; --i) {
            auto temp = residual_norm_collection(i, k);
            for (size_type j = i + 1; j < final_iter_nums[k]; ++j) {
                temp -=
                    hessenberg(i, j * residual_norm_collection.size[1] + k) *
                    y(j, k);
            }
            y(i, k) =
                temp / hessenberg(i, i * residual_norm_collection.size[1] + k);
        }
    }
}


template <typename ValueType, typename ConstAccessor3d>
void calculate_qy(ConstAccessor3d krylov_bases,
                  matrix::view::dense<const ValueType> y,
                  matrix::view::dense<ValueType> before_preconditioner,
                  const size_type* final_iter_nums)
{
    static_assert(
        std::is_same<
            ValueType,
            typename ConstAccessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    for (size_type k = 0; k < before_preconditioner.size[1]; ++k) {
        for (size_type i = 0; i < before_preconditioner.size[0]; ++i) {
            before_preconditioner(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner(i, k) += krylov_bases(j, i, k) * y(j, k);
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ValueType> b,
                matrix::view::dense<ValueType> residual,
                matrix::view::dense<ValueType> givens_sin,
                matrix::view::dense<ValueType> givens_cos,
                array<stopping_status>* stop_status, size_type krylov_dim)
{
    for (size_type j = 0; j < b.size[1]; ++j) {
        for (size_type i = 0; i < b.size[0]; ++i) {
            residual(i, j) = b(i, j);
        }
        for (size_type i = 0; i < krylov_dim; ++i) {
            givens_sin(i, j) = zero<ValueType>();
            givens_cos(i, j) = zero<ValueType>();
        }
        stop_status->get_data()[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_BASE(
    GKO_DECLARE_CB_GMRES_INITIALIZE_KERNEL);


template <typename ValueType, typename Accessor3d>
void restart(std::shared_ptr<const ReferenceExecutor> exec,
             matrix::view::dense<const ValueType> residual,
             matrix::view::dense<remove_complex<ValueType>> residual_norm,
             matrix::view::dense<ValueType> residual_norm_collection,
             matrix::view::dense<remove_complex<ValueType>> arnoldi_norm,
             Accessor3d krylov_bases,
             matrix::view::dense<ValueType> next_krylov_basis,
             array<size_type>* final_iter_nums, array<char>&,
             size_type krylov_dim)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    using rc_vtype = remove_complex<ValueType>;
    constexpr bool has_scalar =
        gko::cb_gmres::detail::has_3d_scaled_accessor<Accessor3d>::value;

    for (size_type j = 0; j < residual.size[1]; ++j) {
        // Calculate residual norm
        residual_norm(0, j) = zero<rc_vtype>();
        if (has_scalar) {
            arnoldi_norm(2, j) = zero<rc_vtype>();
        }
        for (size_type i = 0; i < residual.size[0]; ++i) {
            residual_norm(0, j) += squared_norm(residual(i, j));
            if (has_scalar) {
                arnoldi_norm(2, j) = (arnoldi_norm(2, j) >= abs(residual(i, j)))
                                         ? arnoldi_norm(2, j)
                                         : abs(residual(i, j));
            }
        }
        residual_norm(0, j) = sqrt(residual_norm(0, j));
        gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
            krylov_bases, {0}, j, arnoldi_norm(2, j) / residual_norm(0, j));

        for (size_type i = 0; i < krylov_dim + 1; ++i) {
            if (i == 0) {
                residual_norm_collection(i, j) = residual_norm(0, j);
            } else {
                residual_norm_collection(i, j) = zero<ValueType>();
            }
        }
        for (size_type i = 0; i < residual.size[0]; ++i) {
            krylov_bases(0, i, j) = residual(i, j) / residual_norm(0, j);
            next_krylov_basis(i, j) = residual(i, j) / residual_norm(0, j);
        }
        final_iter_nums->get_data()[j] = 0;
    }

    for (size_type k = 1; k < krylov_dim + 1; ++k) {
        for (size_type j = 0; j < residual.size[1]; ++j) {
            gko::cb_gmres::helper_functions_accessor<Accessor3d>::write_scalar(
                krylov_bases, k, j, one<rc_vtype>());
            for (size_type i = 0; i < residual.size[0]; ++i) {
                krylov_bases(k, i, j) = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_TYPE(GKO_DECLARE_CB_GMRES_RESTART_KERNEL);


template <typename ValueType, typename Accessor3d>
void arnoldi(std::shared_ptr<const ReferenceExecutor> exec,
             matrix::view::dense<ValueType> next_krylov_basis,
             matrix::view::dense<ValueType> givens_sin,
             matrix::view::dense<ValueType> givens_cos,
             matrix::view::dense<remove_complex<ValueType>> residual_norm,
             matrix::view::dense<ValueType> residual_norm_collection,
             Accessor3d krylov_bases,
             matrix::view::dense<ValueType> hessenberg_iter,
             matrix::view::dense<ValueType> buffer_iter,
             matrix::view::dense<remove_complex<ValueType>> arnoldi_norm,
             size_type iter, array<size_type>* final_iter_nums,
             const array<stopping_status>* stop_status, array<stopping_status>*,
             array<size_type>*)
{
    static_assert(
        std::is_same<ValueType,
                     typename Accessor3d::accessor::arithmetic_type>::value,
        "ValueType must match arithmetic_type of accessor!");
    for (size_type i = 0; i < final_iter_nums->get_size(); ++i) {
        final_iter_nums->get_data()[i] +=
            (1 - static_cast<size_type>(
                     stop_status->get_const_data()[i].has_stopped()));
    }
    finish_arnoldi_CGS(next_krylov_basis, krylov_bases, hessenberg_iter,
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
void solve_krylov(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> residual_norm_collection,
                  ConstAccessor3d krylov_bases,
                  matrix::view::dense<const ValueType> hessenberg,
                  matrix::view::dense<ValueType> y,
                  matrix::view::dense<ValueType> before_preconditioner,
                  const array<size_type>* final_iter_nums)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums->get_const_data());
    calculate_qy(krylov_bases, y.as_const(), before_preconditioner,
                 final_iter_nums->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_CB_GMRES_CONST_TYPE(
    GKO_DECLARE_CB_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace cb_gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
