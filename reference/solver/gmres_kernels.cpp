/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

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


#include "core/base/array.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/solver/gmres.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace gmres {


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *b_norm,
                  matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *givens_sin,
                  matrix::Dense<ValueType> *givens_cos,
                  Array<size_type> *final_iter_nums,
                  Array<stopping_status> *stop_status, const int max_iter)
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
        for (size_type i = 0; i < max_iter; ++i) {
            givens_sin->at(i, j) = zero<ValueType>();
            givens_cos->at(i, j) = zero<ValueType>();
        }
        final_iter_nums->get_data()[j] = 0;
        stop_status->get_data()[j].reset();
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norms,
                  matrix::Dense<ValueType> *krylov_bases, const int max_iter)
{
    for (size_type j = 0; j < residual->get_size()[1]; ++j) {
        // Calculate residual norm
        residual_norm->at(0, j) = 0;
        for (size_type i = 0; i < residual->get_size()[0]; ++i) {
            residual_norm->at(0, j) += residual->at(i, j) * residual->at(i, j);
        }
        residual_norm->at(0, j) = sqrt(residual_norm->at(0, j));

        for (size_type i = 0; i < max_iter + 1; ++i) {
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
    for (size_type i = 0; i < next_krylov_basis->get_size()[1]; ++i) {
        for (size_type k = 0; k < iter + 1; ++k) {
            if (stop_status->get_const_data()[i].has_stopped()) {
                continue;
            }
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

        // Start apply givens rotation
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

        // Calculate sin and cos
        if (hessenberg_iter->at(iter, i) == zero<ValueType>()) {
            givens_cos->at(iter, i) = zero<ValueType>();
            givens_sin->at(iter, i) = one<ValueType>();
        } else {
            auto hypotenuse = sqrt(hessenberg_iter->at(iter, i) *
                                       hessenberg_iter->at(iter, i) +
                                   hessenberg_iter->at(iter + 1, i) *
                                       hessenberg_iter->at(iter + 1, i));
            givens_cos->at(iter, i) =
                abs(hessenberg_iter->at(iter, i)) / hypotenuse;
            givens_sin->at(iter, i) = givens_cos->at(iter, i) *
                                      hessenberg_iter->at(iter + 1, i) /
                                      hessenberg_iter->at(iter, i);
        }

        hessenberg_iter->at(iter, i) =
            givens_cos->at(iter, i) * hessenberg_iter->at(iter, i) +
            givens_sin->at(iter, i) * hessenberg_iter->at(iter + 1, i);
        hessenberg_iter->at(iter + 1, i) = zero<ValueType>();
        // hessenberg(iter)   = cos(iter)*hessenberg(iter) +
        //                      sin(iter)*hessenberg(iter)
        // hessenberg(iter+1) = 0
        // End apply givens rotation

        // Calculate residual norm
        residual_norms->at(iter + 1, i) =
            -givens_sin->at(iter, i) * residual_norms->at(iter, i);
        residual_norms->at(iter, i) =
            givens_cos->at(iter, i) * residual_norms->at(iter, i);
        residual_norm->at(0, i) =
            abs(residual_norms->at(iter + 1, i)) / b_norm->at(0, i);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *residual_norms,
            matrix::Dense<ValueType> *krylov_bases,
            matrix::Dense<ValueType> *hessenberg, matrix::Dense<ValueType> *y,
            matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums)
{
    // Solve upper triangular.
    for (size_type k = 0; k < residual_norms->get_size()[1]; ++k) {
        for (int i = final_iter_nums->get_const_data()[k] - 1; i >= 0; --i) {
            auto temp = residual_norms->at(i, k);
            for (size_type j = i + 1; j < final_iter_nums->get_const_data()[k];
                 ++j) {
                temp -=
                    hessenberg->at(i, j * residual_norms->get_size()[1] + k) *
                    y->at(j, k);
            }
            y->at(i, k) =
                temp / hessenberg->at(i, i * residual_norms->get_size()[1] + k);
        }
    }

    // Solve x
    for (size_type k = 0; k < x->get_size()[1]; ++k) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < final_iter_nums->get_const_data()[k];
                 ++j) {
                x->at(i, k) +=
                    krylov_bases->at(i, j * x->get_size()[1] + k) * y->at(j, k);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
