/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/gmres_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/solver/gmres.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace gmres {


namespace {


template <typename ValueType>
void finish_arnoldi(matrix::Dense<ValueType> *next_krylov_basis,
                    matrix::Dense<ValueType> *krylov_bases,
                    matrix::Dense<ValueType> *hessenberg_iter,
                    const size_type iter, const stopping_status *stop_status)
{
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type k = 0; k < iter + 1; ++k) {
            hessenberg_iter->at(k, i) = 0;
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                hessenberg_iter->at(k, i) +=
                    next_krylov_basis->at(j, i) *
                    krylov_bases->at(j,
                                     next_krylov_basis->get_size()[1] * k + i);
            }
            for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
                next_krylov_basis->at(j, i) -=
                    hessenberg_iter->at(k, i) *
                    krylov_bases->at(j,
                                     next_krylov_basis->get_size()[1] * k + i);
            }
        }
        // for i in 1:iter
        //     hessenberg(iter, i) = next_krylov_basis' * krylov_bases(:, i)
        //     next_krylov_basis  -= hessenberg(iter, i) * krylov_bases(:, i)
        // end

        hessenberg_iter->at(iter + 1, i) = 0;
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            hessenberg_iter->at(iter + 1, i) +=
                next_krylov_basis->at(j, i) * next_krylov_basis->at(j, i);
        }
        hessenberg_iter->at(iter + 1, i) =
            sqrt(hessenberg_iter->at(iter + 1, i));
        // hessenberg(iter, iter + 1) = norm(next_krylov_basis)
        for (size_type j = 0; j < next_krylov_basis->get_size()[0]; ++j) {
            next_krylov_basis->at(j, i) /= hessenberg_iter->at(iter + 1, i);
            krylov_bases->at(j, next_krylov_basis->get_size()[1] * (iter + 1) +
                                    i) = next_krylov_basis->at(j, i);
        }
        // next_krylov_basis /= hessenberg(iter, iter + 1)
        // End of arnoldi
    }
}


template <typename ValueType>
void calculate_sin_and_cos(matrix::Dense<ValueType> *givens_sin,
                           matrix::Dense<ValueType> *givens_cos,
                           matrix::Dense<ValueType> *hessenberg_iter,
                           const size_type iter, const size_type rhs)
{
    if (hessenberg_iter->at(iter, rhs) == zero<ValueType>()) {
        givens_cos->at(iter, rhs) = zero<ValueType>();
        givens_sin->at(iter, rhs) = one<ValueType>();
    } else {
        auto hypotenuse = sqrt(hessenberg_iter->at(iter, rhs) *
                                   hessenberg_iter->at(iter, rhs) +
                               hessenberg_iter->at(iter + 1, rhs) *
                                   hessenberg_iter->at(iter + 1, rhs));
        givens_cos->at(iter, rhs) =
            abs(hessenberg_iter->at(iter, rhs)) / hypotenuse;
        givens_sin->at(iter, rhs) = givens_cos->at(iter, rhs) *
                                    hessenberg_iter->at(iter + 1, rhs) /
                                    hessenberg_iter->at(iter, rhs);
    }
}


template <typename ValueType>
void givens_rotation(matrix::Dense<ValueType> *next_krylov_basis,
                     matrix::Dense<ValueType> *givens_sin,
                     matrix::Dense<ValueType> *givens_cos,
                     matrix::Dense<ValueType> *hessenberg_iter,
                     const size_type iter, const stopping_status *stop_status)
{
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        for (size_type j = 0; j < iter; ++j) {
            auto temp = givens_cos->at(j, i) * hessenberg_iter->at(j, i) +
                        givens_sin->at(j, i) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j + 1, i) =
                -givens_sin->at(j, i) * hessenberg_iter->at(j, i) +
                givens_cos->at(j, i) * hessenberg_iter->at(j + 1, i);
            hessenberg_iter->at(j, i) = temp;
            // temp             =  cos(j)*hessenberg(j) +
            //                     sin(j)*hessenberg(j+1)
            // hessenberg(j+1)  = -sin(j)*hessenberg(j) +
            //                     cos(j)*hessenberg(j+1)
            // hessenberg(j)    =  temp;
        }

        calculate_sin_and_cos(givens_sin, givens_cos, hessenberg_iter, iter, i);

        hessenberg_iter->at(iter, i) =
            givens_cos->at(iter, i) * hessenberg_iter->at(iter, i) +
            givens_sin->at(iter, i) * hessenberg_iter->at(iter + 1, i);
        hessenberg_iter->at(iter + 1, i) = zero<ValueType>();
        // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
        //                      sin(iter)*hessenberg(iter)
        // hessenberg(iter+1) = 0
    }
}


template <typename ValueType>
void calculate_next_residual_norm(matrix::Dense<ValueType> *givens_sin,
                                  matrix::Dense<ValueType> *givens_cos,
                                  matrix::Dense<ValueType> *residual_norm,
                                  matrix::Dense<ValueType> *residual_norms,
                                  const matrix::Dense<ValueType> *b_norm,
                                  const size_type iter,
                                  const stopping_status *stop_status)
{
    for (size_type i = 0; i < residual_norm->get_size()[1]; ++i) {
        if (stop_status[i].has_stopped()) {
            continue;
        }
        residual_norms->at(iter + 1, i) =
            -givens_sin->at(iter, i) * residual_norms->at(iter, i);
        residual_norms->at(iter, i) =
            givens_cos->at(iter, i) * residual_norms->at(iter, i);
        residual_norm->at(0, i) =
            abs(residual_norms->at(iter + 1, i)) / b_norm->at(0, i);
    }
}


template <typename ValueType>
void solve_upper_triangular(const matrix::Dense<ValueType> *residual_norms,
                            matrix::Dense<ValueType> *hessenberg,
                            matrix::Dense<ValueType> *y,
                            const size_type *final_iter_nums)
{
    for (size_type k = 0; k < residual_norms->get_size()[1]; ++k) {
        for (int i = final_iter_nums[k] - 1; i >= 0; --i) {
            auto temp = residual_norms->at(i, k);
            for (size_type j = i + 1; j < final_iter_nums[k]; ++j) {
                temp -=
                    hessenberg->at(i, j * residual_norms->get_size()[1] + k) *
                    y->at(j, k);
            }
            y->at(i, k) =
                temp / hessenberg->at(i, i * residual_norms->get_size()[1] + k);
        }
    }
}


template <typename ValueType>
void solve_x(matrix::Dense<ValueType> *krylov_bases,
             matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
             const size_type *final_iter_nums, const LinOp *preconditioner)
{
    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(x);

    for (size_type k = 0; k < x->get_size()[1]; ++k) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            before_preconditioner->at(i, k) = zero<ValueType>();
            for (size_type j = 0; j < final_iter_nums[k]; ++j) {
                before_preconditioner->at(i, k) +=
                    krylov_bases->at(i, j * x->get_size()[1] + k) * y->at(j, k);
            }
        }
        preconditioner->apply(before_preconditioner.get(),
                              after_preconditioner.get());
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            x->at(i, k) += after_preconditioner->at(i, k);
        }
    }
}


}  // namespace


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<stopping_status> *stop_status,
                  const size_type krylov_dim)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        // Calculate b norm
        b_norm->at(0, j) = zero<ValueType>();
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            b_norm->at(0, j) += b->at(i, j) * b->at(i, j);
        }
        b_norm->at(0, j) = sqrt(b_norm->at(0, j));

        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            residual->at(i, j) = b->at(i, j);
        }
        for (size_type i = 0; i < krylov_dim; ++i) {
            givens_sin->at(i, j) = zero<ValueType>();
            givens_cos->at(i, j) = zero<ValueType>();
        }
        stop_status->get_data()[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norms,
                  matrix::Dense<ValueType> *krylov_bases,
                  Array<size_type> *final_iter_nums, const size_type krylov_dim)
{
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        // Calculate residual norm
        residual_norm->at(0, j) = 0;
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            residual_norm->at(0, j) += residual->at(i, j) * residual->at(i, j);
        }
        residual_norm->at(0, j) = sqrt(residual_norm->at(0, j));

        for (size_type i = 0; i < krylov_dim + 1; ++i) {
            if (i == 0) {
                residual_norms->at(i, j) = residual_norm->at(0, j);
            } else {
                residual_norms->at(i, j) = zero<ValueType>();
            }
        }
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            krylov_bases->at(i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
        }
        final_iter_nums->get_data()[j] = 0;
    }

    for (size_type j = residual->get_size()[1]; j < krylov_bases->get_size()[1];
         ++j) {
        for (size_type i = 0; i < krylov_bases->get_size()[0]; ++i) {
            krylov_bases->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *next_krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, const size_type iter,
            const Array<stopping_status> *stop_status)
{
    finish_arnoldi(next_krylov_basis, krylov_bases, hessenberg_iter, iter,
                   stop_status->get_const_data());
    givens_rotation(next_krylov_basis, givens_sin, givens_cos, hessenberg_iter,
                    iter, stop_status->get_const_data());
    calculate_next_residual_norm(givens_sin, givens_cos, residual_norm,
                                 residual_norms, b_norm, iter,
                                 stop_status->get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums,
            const LinOp *preconditioner)
{
    solve_upper_triangular(residual_norms, hessenberg, y,
                           final_iter_nums->get_const_data());
    solve_x(krylov_bases, y, x, final_iter_nums->get_const_data(),
            preconditioner);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
