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


template <typename ValueType, typename AccessorType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *residual,
                  matrix::Dense<ValueType> *residual_norm,
                  matrix::Dense<ValueType> *residual_norms,
                  AccessorType range_Krylov_bases, const int max_iter)
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
            range_Krylov_bases(i, j) =
                residual->at(i, j) / residual_norm->at(0, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType, typename AccessorType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *Krylov_basis,
            matrix::Dense<ValueType> *givens_sin,
            matrix::Dense<ValueType> *givens_cos,
            matrix::Dense<ValueType> *residual_norm,
            matrix::Dense<ValueType> *residual_norms,
            AccessorType range_Krylov_bases, AccessorType range_Hessenberg_iter,
            const matrix::Dense<ValueType> *b_norm, const size_type iter_id,
            const Array<stopping_status> *stop_status)
{
    for (size_type i = 0; i < Krylov_basis->get_size()[1]; ++i) {
        for (size_type k = 0; k < iter_id + 1; ++k) {
            if (stop_status->get_const_data()[i].has_stopped()) {
                continue;
            }
            range_Hessenberg_iter(k, i) = 0;
            for (size_type j = 0; j < Krylov_basis->get_size()[0]; ++j) {
                range_Hessenberg_iter(k, i) +=
                    Krylov_basis->at(j, i) *
                    range_Krylov_bases(j, Krylov_basis->get_size()[1] * k + i);
            }
            for (size_type j = 0; j < Krylov_basis->get_size()[0]; ++j) {
                Krylov_basis->at(j, i) -=
                    range_Hessenberg_iter(k, i) *
                    range_Krylov_bases(j, Krylov_basis->get_size()[1] * k + i);
            }
        }
        range_Hessenberg_iter(iter_id + 1, i) = 0;
        for (size_type j = 0; j < Krylov_basis->get_size()[0]; ++j) {
            range_Hessenberg_iter(iter_id + 1, i) +=
                Krylov_basis->at(j, i) * Krylov_basis->at(j, i);
        }
        range_Hessenberg_iter(iter_id + 1, i) =
            sqrt(range_Hessenberg_iter(iter_id + 1, i));
        for (size_type j = 0; j < Krylov_basis->get_size()[0]; ++j) {
            Krylov_basis->at(j, i) /= range_Hessenberg_iter(iter_id + 1, i);
            range_Krylov_bases(j, Krylov_basis->get_size()[1] * (iter_id + 1) +
                                      i) = Krylov_basis->at(j, i);
        }
        // End of arnoldi

        // Start apply givens rotation
        for (size_type j = 0; j < iter_id; ++j) {
            auto temp = givens_cos->at(j, i) * range_Hessenberg_iter(j, i) +
                        givens_sin->at(j, i) * range_Hessenberg_iter(j + 1, i);
            range_Hessenberg_iter(j + 1, i) =
                -givens_sin->at(j, i) * range_Hessenberg_iter(j, i) +
                givens_cos->at(j, i) * range_Hessenberg_iter(j + 1, i);
            range_Hessenberg_iter(j, i) = temp;
        }
        if (range_Hessenberg_iter(iter_id, i) == zero<ValueType>()) {
            givens_cos->at(iter_id, i) = zero<ValueType>();
            givens_sin->at(iter_id, i) = one<ValueType>();
        } else {
            auto t = sqrt(range_Hessenberg_iter(iter_id, i) *
                              range_Hessenberg_iter(iter_id, i) +
                          range_Hessenberg_iter(iter_id + 1, i) *
                              range_Hessenberg_iter(iter_id + 1, i));
            givens_cos->at(iter_id, i) =
                abs(range_Hessenberg_iter(iter_id, i)) / t;
            givens_sin->at(iter_id, i) = givens_cos->at(iter_id, i) *
                                         range_Hessenberg_iter(iter_id + 1, i) /
                                         range_Hessenberg_iter(iter_id, i);
        }
        range_Hessenberg_iter(iter_id, i) =
            givens_cos->at(iter_id, i) * range_Hessenberg_iter(iter_id, i) +
            givens_sin->at(iter_id, i) * range_Hessenberg_iter(iter_id + 1, i);
        range_Hessenberg_iter(iter_id + 1, i) = zero<ValueType>();
        // End apply givens rotation

        residual_norms->at(iter_id + 1, i) =
            -givens_sin->at(iter_id, i) * residual_norms->at(iter_id, i);
        residual_norms->at(iter_id, i) =
            givens_cos->at(iter_id, i) * residual_norms->at(iter_id, i);
        residual_norm->at(0, i) =
            abs(residual_norms->at(iter_id + 1, i)) / b_norm->at(0, i);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType, typename AccessorType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *residual_norms,
            AccessorType range_Krylov_bases, AccessorType range_Hessenberg,
            matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
            const Array<size_type> *final_iter_nums)
{
    // Solve upper triangular.
    for (size_type k = 0; k < residual_norms->get_size()[1]; ++k) {
        for (int i = final_iter_nums->get_const_data()[k] - 1; i >= 0; --i) {
            auto temp = residual_norms->at(i, k);
            for (size_type j = i + 1; j < final_iter_nums->get_const_data()[k];
                 ++j) {
                temp -=
                    range_Hessenberg(i, j * residual_norms->get_size()[1] + k) *
                    y->at(j, k);
            }
            y->at(i, k) = temp / range_Hessenberg(
                                     i, i * residual_norms->get_size()[1] + k);
        }
    }

    // Solve x
    for (size_type k = 0; k < x->get_size()[1]; ++k) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < final_iter_nums->get_const_data()[k];
                 ++j) {
                x->at(i, k) += range_Krylov_bases(i, j * x->get_size()[1] + k) *
                               y->at(j, k);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
