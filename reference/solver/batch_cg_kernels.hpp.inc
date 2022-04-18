/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const PrecType& prec,
    const gko::batch_dense::BatchEntry<ValueType>& z_entry,
    const gko::batch_dense::BatchEntry<ValueType>& rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType>& p_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        rhs_norms_entry)
{
    // Compute norms of rhs
    compute_norm2<ValueType>(b_entry, rhs_norms_entry);


    // r = b
    copy(b_entry, r_entry);

    // r = b - A*x
    advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_entry,
                           gko::batch::to_const(x_entry),
                           static_cast<ValueType>(1.0), r_entry);
    // z = precond * r
    prec.apply(gko::batch::to_const(r_entry), z_entry);


    // p = z
    copy(gko::batch::to_const(z_entry), p_entry);

    // rho_old = r' * z
    compute_dot_product(gko::batch::to_const(r_entry),
                        gko::batch::to_const(z_entry), rho_old_entry);
}


template <typename ValueType>
inline void update_p(
    const gko::batch_dense::BatchEntry<const ValueType>& rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& z_entry,
    const gko::batch_dense::BatchEntry<ValueType>& p_entry)
{
    const ValueType beta = rho_new_entry.values[0] / rho_old_entry.values[0];
    for (int r = 0; r < p_entry.num_rows; r++) {
        p_entry.values[r * p_entry.stride] =
            z_entry.values[r * z_entry.stride] +
            beta * p_entry.values[r * p_entry.stride];
    }
}


template <typename ValueType>
inline void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType>& rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& p_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& Ap_entry,
    const gko::batch_dense::BatchEntry<ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry)
{
    compute_dot_product<ValueType>(p_entry, Ap_entry, alpha_entry);

    const ValueType alpha = rho_old_entry.values[0] / alpha_entry.values[0];
    for (int r = 0; r < r_entry.num_rows; r++) {
        x_entry.values[r * x_entry.stride] +=
            alpha * p_entry.values[r * p_entry.stride];
        r_entry.values[r * r_entry.stride] -=
            alpha * Ap_entry.values[r * Ap_entry.stride];
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_cg_impl(
    const gko::kernels::batch_cg::BatchCgOptions<remove_complex<ValueType>>&
        opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_type ibatch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    using byte = unsigned char;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;

    byte* const shared_space = local_space;
    ValueType* const r = reinterpret_cast<ValueType*>(shared_space);
    ValueType* const z = r + nrows * nrhs;
    ValueType* const p = z + nrows * nrhs;
    ValueType* const Ap = p + nrows * nrhs;
    ValueType* const prec_work = Ap + nrows * nrhs;
    ValueType* const rho_old =
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz);
    ValueType* const rho_new = rho_old + nrhs;
    ValueType* const alpha = rho_new + nrhs;
    real_type* const norms_rhs = reinterpret_cast<real_type*>(alpha + nrhs);
    real_type* const norms_res = norms_rhs + nrhs;

    uint32 converged = 0;

    const auto A_entry = gko::batch::batch_entry(a, ibatch);
    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(b, ibatch);
    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);

    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        r, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> z_entry{
        z, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> p_entry{
        p, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> Ap_entry{
        Ap, static_cast<size_type>(nrhs), nrows, nrhs};

    const gko::batch_dense::BatchEntry<ValueType> rho_old_entry{
        rho_old, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> rho_new_entry{
        rho_new, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> alpha_entry{
        alpha, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};

    // generate preconditioner
    prec.generate(ibatch, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry, prec,
               z_entry, rho_old_entry, p_entry, rhs_norms_entry);

    // stopping criterion object
    StopType stop(opts.residual_tol, rhs_norms_entry.values);

    int iter{};

    for (iter = 0; iter < opts.max_its; iter++) {
        // use implicit residual norms
        res_norms_entry.values[0] = sqrt(abs(rho_old_entry.values[0]));
        // std::cout << "  Batch " << ibatch << ", Iter " << iter <<
        //    ", res = " << res_norms_entry.values[0] << std::endl;

        if (stop.check_converged(res_norms_entry.values)) {
            break;
        }

        // Ap = A * p
        matvec_kernel(A_entry, gko::batch::to_const(p_entry), Ap_entry);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r(
            gko::batch::to_const(rho_old_entry), gko::batch::to_const(p_entry),
            gko::batch::to_const(Ap_entry), alpha_entry, x_entry, r_entry);

        // z = precond * r
        prec.apply(gko::batch::to_const(r_entry), z_entry);

        // rho_new =  (r)' * (z)
        compute_dot_product<ValueType>(gko::batch::to_const(r_entry),
                                       gko::batch::to_const(z_entry),
                                       rho_new_entry);

        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(gko::batch::to_const(rho_new_entry),
                 gko::batch::to_const(rho_old_entry),
                 gko::batch::to_const(z_entry), p_entry);

        // rho_old = rho_new
        copy(gko::batch::to_const(rho_new_entry), rho_old_entry);
    }

    logger.log_iteration(ibatch, iter, res_norms_entry.values[0]);
}
