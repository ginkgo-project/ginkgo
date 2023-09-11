/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include "core/solver/common_gmres_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/solver/cb_gmres_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The common GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace common_gmres {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                stopping_status* stop_status)
{
    const auto krylov_dim = givens_sin->get_size()[0];
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto b, auto residual, auto givens_sin,
                      auto givens_cos, auto stop_status, auto krylov_dim,
                      auto num_rows) {
            using value_type = std::decay_t<decltype(b(0, 0))>;
            if (i == 0) {
                stop_status[j].reset();
            }
            if (i < num_rows) {
                residual(i, j) = b(i, j);
            }
            if (i < krylov_dim) {
                givens_sin(i, j) = zero<value_type>();
                givens_cos(i, j) = zero<value_type>();
            }
        },
        dim<2>{std::max(b->get_size()[0], krylov_dim), b->get_size()[1]}, b,
        residual, givens_sin, givens_cos, stop_status, krylov_dim,
        b->get_size()[0]);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMMON_GMRES_INITIALIZE_KERNEL);


template <typename ValueType>
void hessenberg_qr(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Dense<ValueType>* givens_sin,
                   matrix::Dense<ValueType>* givens_cos,
                   matrix::Dense<remove_complex<ValueType>>* residual_norm,
                   matrix::Dense<ValueType>* residual_norm_collection,
                   matrix::Dense<ValueType>* hessenberg_iter, size_type iter,
                   size_type* final_iter_nums,
                   const stopping_status* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto rhs, auto givens_sin, auto givens_cos,
                      auto residual_norm, auto residual_norm_collection,
                      auto hessenberg_iter, auto iter, auto final_iter_nums,
                      auto stop_status) {
            using value_type = std::decay_t<decltype(givens_sin(0, 0))>;
            if (stop_status[rhs].has_stopped()) {
                return;
            }
            // increment iteration count
            final_iter_nums[rhs]++;
            auto hess_this = hessenberg_iter(0, rhs);
            auto hess_next = hessenberg_iter(1, rhs);
            // apply previous Givens rotations to column
            for (decltype(iter) j = 0; j < iter; ++j) {
                // in here: hess_this = hessenberg_iter(j, rhs);
                //          hess_next = hessenberg_iter(j+1, rhs);
                hess_next = hessenberg_iter(j + 1, rhs);
                const auto gc = givens_cos(j, rhs);
                const auto gs = givens_sin(j, rhs);
                const auto out1 = gc * hess_this + gs * hess_next;
                const auto out2 = conj(gc) * hess_next - conj(gs) * hess_this;
                hessenberg_iter(j, rhs) = out1;
                hessenberg_iter(j + 1, rhs) = hess_this = out2;
                hess_next = hessenberg_iter(j + 2, rhs);
            }
            // hess_this is hessenberg_iter(iter, rhs) and
            // hess_next is hessenberg_iter(iter + 1, rhs)
            auto gs = givens_sin(iter, rhs);
            auto gc = givens_cos(iter, rhs);
            // compute new Givens rotation
            if (hess_this == zero<value_type>()) {
                givens_cos(iter, rhs) = gc = zero<value_type>();
                givens_sin(iter, rhs) = gs = one<value_type>();
            } else {
                const auto scale = abs(hess_this) + abs(hess_next);
                const auto hypotenuse =
                    scale *
                    sqrt(abs(hess_this / scale) * abs(hess_this / scale) +
                         abs(hess_next / scale) * abs(hess_next / scale));
                givens_cos(iter, rhs) = gc = conj(hess_this) / hypotenuse;
                givens_sin(iter, rhs) = gs = conj(hess_next) / hypotenuse;
            }
            // apply new Givens rotation to column
            hessenberg_iter(iter, rhs) = gc * hess_this + gs * hess_next;
            hessenberg_iter(iter + 1, rhs) = zero<value_type>();
            // apply new Givens rotation to RHS of least-squares problem
            const auto rnc_new = zero<value_type>() -
                                 conj(gs) * residual_norm_collection(iter, rhs);
            residual_norm_collection(iter + 1, rhs) = rnc_new;
            residual_norm_collection(iter, rhs) =
                gc * residual_norm_collection(iter, rhs);
            residual_norm(0, rhs) = abs(rnc_new);
        },
        hessenberg_iter->get_size()[1], givens_sin, givens_cos, residual_norm,
        residual_norm_collection, hessenberg_iter, iter, final_iter_nums,
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_COMMON_GMRES_HESSENBERG_QR_KERNEL);


template <typename ValueType>
void solve_krylov(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y, const size_type* final_iter_nums,
                  const stopping_status* stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto rhs, auto mtx, auto y, auto sizes,
                      auto stop, auto num_cols) {
            if (stop[col].is_finalized()) {
                return;
            }
            for (int64 i = sizes[col] - 1; i >= 0; i--) {
                auto value = rhs(i, col);
                for (int64 j = i + 1; j < sizes[col]; j++) {
                    value -= mtx(i, j * num_cols + col) * y(j, col);
                }
                // y(i) = (rhs(i) - U(i,i+1:) * y(i+1:)) / U(i, i)
                y(i, col) = value / mtx(i, i * num_cols + col);
            }
        },
        residual_norm_collection->get_size()[1], residual_norm_collection,
        hessenberg, y, final_iter_nums, stop_status,
        residual_norm_collection->get_size()[1]);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_COMMON_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace common_gmres
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
