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

template <typename StopType, typename PrecType, typename LogType,
          typename MatrixType, typename ValueType>
__global__ void apply_kernel(const int max_iter,
                             const gko::remove_complex<ValueType> tol,
                             const gko::remove_complex<ValueType> relax_factor,
                             LogType logger, PrecType prec,
                             const MatrixType mat,
                             const ValueType* const __restrict__ b,
                             ValueType* const __restrict__ x)
{
    static_assert(
        std::is_same<typename MatrixType::value_type, const ValueType>::value,
        "Inconsistent value type of matrix type!");
    static_assert(std::is_same<typename PrecType::value_type, ValueType>::value,
                  "Inconsistent value type of preconditioner!");
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nbatch = mat.num_batch;
    const auto num_rows = mat.num_rows;
    constexpr auto num_rhs = 1;

    constexpr auto tile_size = config::warp_size;
    auto thread_block = group::this_thread_block();
    auto warp_grp = group::tiled_partition<tile_size>(thread_block);

    for (size_type ibatch = blockIdx.x; ibatch < nbatch; ibatch += gridDim.x) {
        extern __shared__ char local_mem[];
        const auto residual = reinterpret_cast<ValueType*>(local_mem);
        ValueType* const delta_x = residual + num_rows * num_rhs;
        ValueType* const prec_work = residual + 2 * num_rows * num_rhs;
        __shared__ real_type norms[1];
        __shared__ real_type init_res_norm[1];

        const auto a_b = gko::batch::batch_entry(mat, ibatch);
        const ValueType* const b_b =
            gko::batch::batch_entry_ptr(b, 1, num_rows, ibatch);
        ValueType* const x_b =
            gko::batch::batch_entry_ptr(x, 1, num_rows, ibatch);
        const auto relax = static_cast<ValueType>(relax_factor);

        prec.generate(ibatch, a_b, prec_work);

        // initial residual
        if (threadIdx.x / config::warp_size == 0) {
            single_compute_norm2(warp_grp, num_rows, b_b, init_res_norm[0]);
        }

        StopType stop(tol, init_res_norm);

        int iter = 0;
        for (; iter < max_iter; iter++) {
            // r <- r - Adx  This causes instability!
            // adv_spmv_ker(static_cast<ValueType>(-1.0), a_b,
            //     gko::batch_dense::to_const(dx_b),
            //     static_cast<ValueType>(1.0), r_b);

            // r <- b - Ax
            single_copy(num_rows, b_b, residual);
            __syncthreads();
            single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_b,
                                          x_b, static_cast<ValueType>(1.0),
                                          residual);
            __syncthreads();

            if (threadIdx.x / config::warp_size == 0) {
                single_compute_norm2(warp_grp, num_rows, residual, norms[0]);
            }
            __syncthreads();

            if (stop.check_converged(norms)) {
                break;
            }

            prec.apply(num_rows, residual, delta_x);
            __syncthreads();

            single_add_scaled(num_rows, relax, delta_x, x_b);
            __syncthreads();
        }

        logger.log_iteration(ibatch, iter, norms[0]);
    }
}
