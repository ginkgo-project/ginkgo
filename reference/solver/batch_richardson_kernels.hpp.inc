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
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_richardson_impl(
    const gko::kernels::batch_rich::BatchRichardsonOptions<
        remove_complex<ValueType>>& opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_type ibatch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    GKO_ASSERT((nrhs == x.num_rhs));

    const auto residual = reinterpret_cast<ValueType*>(local_space);
    ValueType* const delta_x = residual + nrows * nrhs;
    ValueType* const prec_work = delta_x + nrows * nrhs;
    real_type norms[max_num_rhs];
    real_type init_res_norm[max_num_rhs];

    const auto a_b = gko::batch::batch_entry(gko::batch::to_const(a), ibatch);
    const auto b_b = gko::batch::batch_entry(gko::batch::to_const(b), ibatch);
    const auto x_b = gko::batch::batch_entry(x, ibatch);
    const gko::batch_dense::BatchEntry<ValueType> r_b{
        residual, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> dx_b{
        delta_x, static_cast<size_type>(nrhs), nrows, nrhs};

    const auto relax = static_cast<ValueType>(opts.relax_factor);
    const gko::batch_dense::BatchEntry<const ValueType> relax_b{&relax, 1, 1,
                                                                1};
    const gko::batch_dense::BatchEntry<real_type> norms_b{
        norms, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> init_res_norm_b{
        init_res_norm, static_cast<size_type>(nrhs), 1, nrhs};

    prec.generate(ibatch, a_b, prec_work);

    // initial residual
    compute_norm2<ValueType>(gko::batch::to_const(b_b), norms_b);
    copy(gko::batch::to_const(norms_b), init_res_norm_b);

    StopType stop(opts.residual_tol, init_res_norm_b.values);

    int iter = 0;
    for (; iter < opts.max_its; iter++) {
        // r <- r - Adx  This causes instability!
        // adv_spmv_ker(static_cast<ValueType>(-1.0), a_b,
        //              gko::batch::to_const(dx_b),
        // static_cast<ValueType>(1.0), r_b);

        // r <- b - Ax
        copy(b_b, r_b);
        advanced_matvec_kernel(static_cast<ValueType>(-1.0), a_b,
                               gko::batch::to_const(x_b),
                               static_cast<ValueType>(1.0), r_b);

        compute_norm2(gko::batch::to_const(r_b), norms_b);

        if (stop.check_converged(norms)) {
            break;
        }

        prec.apply(gko::batch::to_const(r_b), dx_b);

        add_scaled(relax_b, gko::batch::to_const(dx_b), x_b);
    }

    logger.log_iteration(ibatch, iter, norms[0]);
}
