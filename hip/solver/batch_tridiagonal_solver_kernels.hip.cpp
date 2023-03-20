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

#include "core/solver/batch_tridiagonal_solver_kernels.hpp"


#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/exception.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/load_store.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/matrix/batch_struct.hip.hpp"

namespace gko {
namespace kernels {
namespace hip {
namespace batch_tridiagonal_solver {

#define GKO_HIP_WMpGE_BATCH_TRIDIAGONAL_SUBWARP_SIZES_CODE \
    1, 2, 4, 8, 16, 32, 64

using batch_WM_pGE_tridiagonal_solver_hip_compiled_subwarp_sizes =
    syn::value_list<int, GKO_HIP_WMpGE_BATCH_TRIDIAGONAL_SUBWARP_SIZES_CODE>;

namespace {

constexpr int default_block_size = 128;

#include "common/cuda_hip/solver/batch_tridiagonal_solver_kernels.hpp.inc"

}  // namespace

namespace {

template <int compiled_WM_pGE_subwarp_size, typename ValueType>
void WM_pGE_app1_helper(
    syn::value_list<int, compiled_WM_pGE_subwarp_size>,
    const int number_WM_steps, const size_type nbatch, const int nrows,
    const int nrhs, const ValueType* const tridiag_mat_subdiags,
    const ValueType* const tridiag_mat_maindiags,
    ValueType* const tridiag_mat_superdiags, ValueType* const rhs,
    ValueType* const x,
    const enum gko::solver::batch_tridiag_solve_approach approach)
{
    constexpr auto subwarp_size = compiled_WM_pGE_subwarp_size;
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * subwarp_size, default_block_size));

    const int shared_size =
        gko::kernels::batch_tridiagonal_solver::local_memory_requirement<
            ValueType>(nrows, nrhs);

    hipLaunchKernelGGL(
        WM_pGE_kernel_approach_1<subwarp_size>, block, shared_size, 0,
        number_WM_steps, nbatch, nrows, as_hip_type(tridiag_mat_subdiags),
        as_hip_type(tridiag_mat_maindiags), as_hip_type(tridiag_mat_superdiags),
        as_hip_type(rhs), as_hip_type(x));


    GKO_HIP_LAST_IF_ERROR_THROW;
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_WM_pGE_app1_helper,
                                    WM_pGE_app1_helper);

template <int compiled_WM_pGE_subwarp_size, typename ValueType>
void WM_pGE_app2_helper(
    syn::value_list<int, compiled_WM_pGE_subwarp_size>,
    const int number_WM_steps, const size_type nbatch, const int nrows,
    const int nrhs, const ValueType* const tridiag_mat_subdiags,
    const ValueType* const tridiag_mat_maindiags,
    ValueType* const tridiag_mat_superdiags, ValueType* const rhs,
    ValueType* const x,
    const enum gko::solver::batch_tridiag_solve_approach approach)
{
    constexpr auto subwarp_size = compiled_WM_pGE_subwarp_size;
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * subwarp_size, default_block_size));

    const int shared_size =
        gko::kernels::batch_tridiagonal_solver::local_memory_requirement<
            ValueType>(nrows, nrhs);

    using HipValueType = typename gko::kernels::hip::hip_type<ValueType>;

    hipLaunchKernelGGL(
        WM_pGE_kernel_approach_2<HipValueType, subwarp_size>, grid, block,
        shared_size, 0, number_WM_steps, nbatch, nrows,
        as_hip_type(tridiag_mat_subdiags), as_hip_type(tridiag_mat_maindiags),
        as_hip_type(tridiag_mat_superdiags), as_hip_type(rhs), as_hip_type(x));

    GKO_HIP_LAST_IF_ERROR_THROW;
}

template <int compiled_WM_pGE_subwarp_size, typename T>
void WM_pGE_app2_helper(
    syn::value_list<int, compiled_WM_pGE_subwarp_size>,
    const int number_WM_steps, const size_type nbatch, const int nrows,
    const int nrhs, const std::complex<T>* const tridiag_mat_subdiags,
    const std::complex<T>* const tridiag_mat_maindiags,
    std::complex<T>* const tridiag_mat_superdiags, std::complex<T>* const rhs,
    std::complex<T>* const x,
    const enum gko::solver::batch_tridiag_solve_approach approach)
{
    throw std::runtime_error(
        "WM_pGE approach 2 does not yet work with complex data types");
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_WM_pGE_app2_helper,
                                    WM_pGE_app2_helper);

template <typename ValueType>
void perform_workspace_copies(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchTridiagonal<ValueType>* const tridiag_mat,
    const matrix::BatchDense<ValueType>* const rhs, const int workspace_size,
    ValueType* const workspace_ptr, ValueType*& tridiag_mat_superdiags,
    ValueType*& rhs_vals)
{
    assert(workspace_size >=
           tridiag_mat->get_num_stored_elements_per_diagonal() +
               rhs->get_num_stored_elements());

    tridiag_mat_superdiags = workspace_ptr;

    exec->copy(tridiag_mat->get_num_stored_elements_per_diagonal(),
               tridiag_mat->get_const_super_diagonal(), tridiag_mat_superdiags);

    rhs_vals =
        workspace_ptr + tridiag_mat->get_num_stored_elements_per_diagonal();

    exec->copy(rhs->get_num_stored_elements(), rhs->get_const_values(),
               rhs_vals);
}

}  // anonymous namespace


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::BatchTridiagonal<ValueType>* const tridiag_mat,
           const matrix::BatchDense<ValueType>* const rhs,
           matrix::BatchDense<ValueType>* const x, const int workspace_size,
           ValueType* const workspace_ptr, const int number_WM_steps,
           const int user_given_WM_pGE_subwarp_size,
           const enum gko::solver::batch_tridiag_solve_approach approach,
           double& millisec_subtract)
{
    const auto nbatch = tridiag_mat->get_num_batch_entries();
    const auto nrows = static_cast<int>(tridiag_mat->get_size().at(0)[0]);
    const auto nrhs = static_cast<int>(rhs->get_size().at(0)[1]);
    assert(nrhs == 1);

    if (approach == gko::solver::batch_tridiag_solve_approach::WM_pGE_app1) {
        ValueType* tridiag_mat_superdiags;
        ValueType* rhs_vals;

        perform_workspace_copies(exec, tridiag_mat, rhs, workspace_size,
                                 workspace_ptr, tridiag_mat_superdiags,
                                 rhs_vals);

        select_WM_pGE_app1_helper(
            batch_WM_pGE_tridiagonal_solver_hip_compiled_subwarp_sizes(),
            [&](int compiled_WM_pGE_subwarp_size) {
                return user_given_WM_pGE_subwarp_size ==
                       compiled_WM_pGE_subwarp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), number_WM_steps, nbatch,
            nrows, nrhs, tridiag_mat->get_const_sub_diagonal(),
            tridiag_mat->get_const_main_diagonal(), tridiag_mat_superdiags,
            rhs_vals, x->get_values(), approach);

    } else if (approach ==
               gko::solver::batch_tridiag_solve_approach::WM_pGE_app2) {
        ValueType* tridiag_mat_superdiags;
        ValueType* rhs_vals;

        perform_workspace_copies(exec, tridiag_mat, rhs, workspace_size,
                                 workspace_ptr, tridiag_mat_superdiags,
                                 rhs_vals);

        select_WM_pGE_app2_helper(
            batch_WM_pGE_tridiagonal_solver_hip_compiled_subwarp_sizes(),
            [&](int compiled_WM_pGE_subwarp_size) {
                return user_given_WM_pGE_subwarp_size ==
                       compiled_WM_pGE_subwarp_size;
            },
            syn::value_list<int>(), syn::type_list<>(), number_WM_steps, nbatch,
            nrows, nrhs, tridiag_mat->get_const_sub_diagonal(),
            tridiag_mat->get_const_main_diagonal(), tridiag_mat_superdiags,
            rhs_vals, x->get_values(), approach);

    } else if (approach ==
               gko::solver::batch_tridiag_solve_approach::vendor_provided) {
        GKO_NOT_IMPLEMENTED;
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_TRIDIAGONAL_SOLVER_APPLY_KERNEL);

}  // namespace batch_tridiagonal_solver
}  // namespace hip
}  // namespace kernels
}  // namespace gko
