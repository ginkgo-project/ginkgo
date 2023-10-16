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
__dpct_inline__ void apply_kernel(
    const int max_iter, const gko::remove_complex<ValueType> tol,
    const gko::remove_complex<ValueType> relax_factor, LogType logger,
    PrecType prec, const MatrixType a, const ValueType* const __restrict__ b,
    ValueType* const __restrict__ x, const size_type num_rows,
    const size_type num_rhs, ValueType* slm_values,
    gko::remove_complex<ValueType>* slm_reals, sycl::nd_item<3> item_ct1)
{
    static_assert(
        std::is_same<typename MatrixType::value_type, const ValueType>::value,
        "Inconsistent value type of matrix type!");
    static_assert(std::is_same<typename PrecType::value_type, ValueType>::value,
                  "Inconsistent value type of preconditioner!");
    using real_type = typename gko::remove_complex<ValueType>;

    auto sg = item_ct1.get_sub_group();
    const auto sg_id = sg.get_group_id();
    const auto ibatch = item_ct1.get_group_linear_id();

    ValueType* residual = &slm_values[0];
    ValueType* delta_x = residual + num_rows * num_rhs;
    ValueType* prec_work = residual + 2 * num_rows * num_rhs;
    real_type* norms = &slm_reals[0];
    real_type* init_res_norm = &slm_reals[1];

    const auto relax = static_cast<ValueType>(relax_factor);

    prec.generate(ibatch, a, prec_work, item_ct1);

    // initial residual
    if (sg_id == 0) {
        compute_norm2_sg_kernel(num_rows, b, init_res_norm[0], item_ct1);
    }

    StopType stop(tol, init_res_norm);

    int iter = 0;
    for (; iter < max_iter; iter++) {
        // r <- r - Adx  This causes instability!
        // adv_spmv_ker(static_cast<ValueType>(-1.0), a,
        //     gko::batch_dense::to_const(dx_b),
        //     static_cast<ValueType>(1.0), r_b);

        // r <- b - Ax
        copy_kernel(num_rows, b, residual, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
        single_advanced_matvec_kernel(static_cast<ValueType>(-1.0), a, x,
                                      static_cast<ValueType>(1.0), residual,
                                      item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (sg_id == 0) {
            compute_norm2_sg_kernel(num_rows, residual, norms[0], item_ct1);
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        if (stop.check_converged(norms)) {
            break;
        }

        prec.apply(num_rows, residual, delta_x, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);

        add_scaled_kernel(num_rows, relax, delta_x, x, item_ct1);
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }

    logger.log_iteration(ibatch, iter, norms[0]);
}
