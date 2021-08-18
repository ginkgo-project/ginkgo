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


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "ginkgo/core/stop/stopping_status.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The GMRES solver namespace.
 *
 * @ingroup gmres
 */
namespace gmres {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType>* b,
                matrix::Dense<ValueType>* residual,
                matrix::Dense<ValueType>* givens_sin,
                matrix::Dense<ValueType>* givens_cos,
                Array<stopping_status>& stop_status)
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_KERNEL);


template <typename ValueType>
void restart(std::shared_ptr<const DefaultExecutor> exec,
             const matrix::Dense<ValueType>* residual,
             matrix::Dense<remove_complex<ValueType>>* residual_norm,
             matrix::Dense<ValueType>* residual_norm_collection,
             matrix::Dense<ValueType>* krylov_bases,
             Array<size_type>& final_iter_nums)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto residual, auto residual_norm,
                      auto residual_norm_collection, auto krylov_bases,
                      auto final_iter_nums) {
            if (i == 0) {
                residual_norm_collection(0, j) = residual_norm(0, j);
                final_iter_nums[j] = 0;
            }
            krylov_bases(i, j) = residual(i, j) / residual_norm(0, j);
        },
        residual->get_size(), residual, residual_norm, residual_norm_collection,
        krylov_bases, final_iter_nums);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_RESTART_KERNEL);


template <typename ValueType>
void hessenberg_qr(std::shared_ptr<const DefaultExecutor> exec,
                   matrix::Dense<ValueType>* givens_sin,
                   matrix::Dense<ValueType>* givens_cos,
                   matrix::Dense<remove_complex<ValueType>>* residual_norm,
                   matrix::Dense<ValueType>* residual_norm_collection,
                   matrix::Dense<ValueType>* hessenberg_iter, size_type iter,
                   Array<size_type>& final_iter_nums,
                   const Array<stopping_status>& stop_status)
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
            // apply previous Givens rotations to column
            for (int64 j = 0; j < iter; ++j) {
                auto out1 = givens_cos(j, rhs) * hessenberg_iter(j, rhs) +
                            givens_sin(j, rhs) * hessenberg_iter(j + 1, rhs);
                auto out2 =
                    -conj(givens_sin(j, rhs)) * hessenberg_iter(j, rhs) +
                    conj(givens_cos(j, rhs)) * hessenberg_iter(j + 1, rhs);
                hessenberg_iter(j, rhs) = out1;
                hessenberg_iter(j + 1, rhs) = out2;
            }
            // compute new Givens rotation
            if (hessenberg_iter(iter, rhs) == zero<value_type>()) {
                givens_cos(iter, rhs) = zero<value_type>();
                givens_sin(iter, rhs) = one<value_type>();
            } else {
                const auto this_hess = hessenberg_iter(iter, rhs);
                const auto next_hess = hessenberg_iter(iter + 1, rhs);
                const auto scale = abs(this_hess) + abs(next_hess);
                const auto hypotenuse =
                    scale *
                    sqrt(abs(this_hess / scale) * abs(this_hess / scale) +
                         abs(next_hess / scale) * abs(next_hess / scale));
                givens_cos(iter, rhs) = conj(this_hess) / hypotenuse;
                givens_sin(iter, rhs) = conj(next_hess) / hypotenuse;
            }
            // apply new Givens rotation to column
            hessenberg_iter(iter, rhs) =
                givens_cos(iter, rhs) * hessenberg_iter(iter, rhs) +
                givens_sin(iter, rhs) * hessenberg_iter(iter + 1, rhs);
            hessenberg_iter(iter + 1, rhs) = zero<value_type>();
            // apply new Givens rotation to RHS of least-squares problem
            residual_norm_collection(iter + 1, rhs) =
                -conj(givens_sin(iter, rhs)) *
                residual_norm_collection(iter, rhs);
            residual_norm_collection(iter, rhs) =
                givens_cos(iter, rhs) * residual_norm_collection(iter, rhs);
            residual_norm(0, rhs) =
                abs(residual_norm_collection(iter + 1, rhs));
        },
        hessenberg_iter->get_size()[1], givens_sin, givens_cos, residual_norm,
        residual_norm_collection, hessenberg_iter, iter, final_iter_nums,
        stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_HESSENBERG_QR_KERNEL);


template <typename ValueType>
void solve_krylov(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* residual_norm_collection,
                  const matrix::Dense<ValueType>* krylov_bases,
                  const matrix::Dense<ValueType>* hessenberg,
                  matrix::Dense<ValueType>* y,
                  matrix::Dense<ValueType>* before_preconditioner,
                  const Array<size_type>& final_iter_nums,
                  Array<stopping_status>& stop_status)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto rhs, auto mtx, auto y, auto sizes,
                      auto stop, auto num_cols) {
            if (stop[col].is_finalized()) {
                return;
            }
            for (int i = sizes[col] - 1; i >= 0; i--) {
                auto value = rhs(i, col);
                for (int j = i + 1; j < sizes[col]; j++) {
                    value -= mtx(i, j * num_cols + col) * y(j, col);
                }
                // y(i) = (rhs(i) - U(i,i+1:) * y(i+1:)) / U(i, i)
                y(i, col) = value / mtx(i, i * num_cols + col);
            }
        },
        residual_norm_collection->get_size()[1], residual_norm_collection,
        hessenberg, y, final_iter_nums, stop_status,
        residual_norm_collection->get_size()[1]);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto col, auto bases, auto y, auto out,
                      auto sizes, auto stop, auto num_rows) {
            if (stop[col].is_finalized()) {
                return;
            }
            auto value = zero(out(row, col));
            for (int i = 0; i < sizes[col]; i++) {
                value += bases(row + i * num_rows, col) * y(i, col);
            }
            out(row, col) = value;
        },
        before_preconditioner->get_size(), krylov_bases, y,
        before_preconditioner, final_iter_nums, stop_status,
        before_preconditioner->get_size()[0]);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto col, auto stop) {
            if (!stop[col].is_finalized()) {
                stop[col].finalize();
            }
        },
        stop_status.get_num_elems(), stop_status);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_SOLVE_KRYLOV_KERNEL);


}  // namespace gmres
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
