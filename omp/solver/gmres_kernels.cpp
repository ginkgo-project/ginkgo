/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


namespace {


template <typename ValueType>
void finish_arnoldi(size_type num_rows, matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                    const stopping_status *stop_status)
{
    const auto krylov_bases_rowoffset = num_rows;
    const auto next_krylov_rowoffset = (iter + 1) * krylov_bases_rowoffset;
#pragma omp declare reduction(add:ValueType : omp_out = omp_out + omp_in)
    for (size_type i = 0; i < hessenberg_iter->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type k = 0; k < iter + 1; ++k) {
            ValueType hessenberg_iter_entry = zero<ValueType>();

#pragma omp parallel for reduction(add : hessenberg_iter_entry)
            for (size_type j = 0; j < num_rows; ++j) {
                hessenberg_iter_entry +=
                    krylov_bases->at(j + next_krylov_rowoffset, i) *
                    krylov_bases->at(j + k * krylov_bases_rowoffset, i);
            }
            hessenberg_iter->at(k, i) = hessenberg_iter_entry;
#pragma omp parallel for
            for (size_type j = 0; j < num_rows; ++j) {
                krylov_bases->at(j + next_krylov_rowoffset, i) -=
                    hessenberg_iter->at(k, i) *
                    krylov_bases->at(j + k * krylov_bases_rowoffset, i);
            }
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end

        ValueType hessenberg_iter_entry = zero<ValueType>();

#pragma omp parallel for reduction(add : hessenberg_iter_entry)
        for (size_type j = 0; j < num_rows; ++j) {
            hessenberg_iter_entry +=
                krylov_bases->at(j + next_krylov_rowoffset, i) *
                krylov_bases->at(j + next_krylov_rowoffset, i);
        }
        hessenberg_iter->at(iter + 1, i) = sqrt(hessenberg_iter_entry);
// hessenberg(iter + 1, iter) = norm(krylov_bases)
#pragma omp parallel for
        for (size_type j = 0; j < num_rows; ++j) {
            krylov_bases->at(j + next_krylov_rowoffset, i) /=
                hessenberg_iter->at(iter + 1, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // End of arnoldi
    }
}


template <typename ValueType>
void calculate_sin_and_cos(matrix::Dense<ValueType> *givens_sin,
                           matrix::Dense<ValueType> *givens_cos,
                           matrix::Dense<ValueType> *hessenberg_iter,
                           size_type iter, const size_type rhs)
{
    if (hessenberg_iter->at(iter, rhs) == zero<ValueType>()) {
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
void givens_rotation(matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
                     const stopping_status *stop_status)
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
    matrix::Dense<ValueType> *givens_sin, matrix::Dense<ValueType> *givens_cos,
    matrix::Dense<remove_complex<ValueType>> *residual_norm,
    matrix::Dense<ValueType> *residual_norm_collection, size_type iter,
    const stopping_status *stop_status)
{
#pragma omp parallel for
    for (size_type i = 0; i < residual_norm->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        residual_norm_collection->at(iter + 1, i) =
            -conj(givens_sin)->at(iter, i) *
            residual_norm_collection->at(iter, i);
        residual_norm_collection->at(iter, i) =
            givens_cos->at(iter, i) * residual_norm_collection->at(iter, i);
        residual_norm->at(0, i) =
            abs(residual_norm_collection->at(iter + 1, i));
    }
}


template <typename ValueType>
void solve_upper_triangular(
    const matrix::Dense<ValueType> *residual_norm_collection,
    const matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
    const size_type *final_iter_nums)
{
#pragma omp parallel for
    for (size_type k = 0; k < residual_norm_collection->get_size()[1]; ++k) {
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


template <typename ValueType>
void calculate_qy(const matrix::Dense<ValueType> *krylov_bases,
                  const matrix::Dense<ValueType> *y,
                  matrix::Dense<ValueType> *before_preconditioner,
                  const size_type *final_iter_nums)
{
    const auto krylov_bases_rowoffset = before_preconditioner->get_size()[0];
#pragma omp parallel for
    for (size_type i = 0; i < before_preconditioner->get_size()[0]; ++i) {
        for (size_type k = 0; k < before_preconditioner->get_size()[1]; ++k) {
            before_preconditioner->at(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner->at(i, k) +=
                    krylov_bases->at(i + j * krylov_bases_rowoffset, k) *
                    y->at(j, k);
            }
        }
    }
}


}  // namespace


template <typename ValueType>
void initialize_1(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status, size_type krylov_dim)
{
    using norm_type = remove_complex<ValueType>;
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<remove_complex<ValueType>> *residual_norm,
                  matrix::Dense<ValueType> *residual_norm_collection,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, size_type krylov_dim)
{
    using norm_type = remove_complex<ValueType>;
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        // Calculate residual norm
        norm_type res_norm = zero<norm_type>();
#pragma omp declare reduction(add:norm_type : omp_out = omp_out + omp_in)

#pragma omp parallel for reduction(add : res_norm)
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            res_norm += squared_norm(residual->at(i, j));
        }
        residual_norm->at(0, j) = sqrt(res_norm);
        residual_norm_collection->at(0, j) = residual_norm->at(0, j);
#pragma omp parallel for
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            krylov_bases->at(i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
        }
        final_iter_nums->get_data()[j] = 0;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const OmpExecutor> exec, size_type num_rows,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<remove_complex<ValueType>> *residual_norm,
            matrix::Dense<ValueType> *residual_norm_collection,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter, size_type iter,
            Array<size_type> *final_iter_nums,
            const Array<stopping_status> *stop_status)
{
#pragma omp parallel for
    for (size_type i = 0; i < final_iter_nums->get_num_elems(); ++i) {
        final_iter_nums->get_data()[i] +=
            (1 - stop_status->get_const_data()[i].has_stopped());
    }

    finish_arnoldi(num_rows, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(givens_sin, givens_cos, hessenberg_iter, iter,
                    stop_status->get_const_data());
    calculate_next_residual_norm(givens_sin, givens_cos, residual_norm,
                                 residual_norm_collection, iter,
                                 stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const OmpExecutor> exec,
            const matrix::Dense<ValueType> *residual_norm_collection,
            const matrix::Dense<ValueType> *krylov_bases,
            const matrix::Dense<ValueType> *hessenberg,
            matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *before_preconditioner,
            const Array<size_type> *final_iter_nums)
{
    solve_upper_triangular(residual_norm_collection, hessenberg, y,
                           final_iter_nums->get_const_data());
    calculate_qy(krylov_bases, y, before_preconditioner,
                 final_iter_nums->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace omp
}  // namespace kernels
}  // namespace gko
