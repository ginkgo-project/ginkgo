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


for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
    byte *const shared_space = local_space.get_data();
    ValueType *const r = reinterpret_cast<ValueType *>(shared_space);
    ValueType *const z = r + nrows * nrhs;
    ValueType *const p = z + nrows * nrhs;
    ValueType *const Ap = p + nrows * nrhs;
    ValueType *const prec_work = Ap + nrows * nrhs;
    ValueType *const rho_old =
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz);
    ValueType *const rho_new = rho_old + nrhs;
    ValueType *const alpha = rho_new + nrhs;
    real_type *const norms_rhs = reinterpret_cast<real_type *>(alpha + nrhs);
    real_type *const norms_res = norms_rhs + nrhs;

    uint32 converged = 0;


    const gko::batch_dense::BatchEntry<const ValueType> left_entry =
        gko::batch::batch_entry(left, ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> right_entry =
        gko::batch::batch_entry(right, ibatch);

    // scale the matrix and rhs
    if (left_entry.values) {
        const typename BatchMatrixType::entry_type A_entry =
            gko::batch::batch_entry(a, ibatch);
        const gko::batch_dense::BatchEntry<ValueType> b_entry =
            gko::batch::batch_entry(b, ibatch);
        batch_scale(left_entry, right_entry, A_entry);
        batch_dense::batch_scale(left_entry, b_entry);
    }

    // const typename BatchMatrixType::entry_type A_entry =
    const auto A_entry =
        gko::batch::batch_entry(gko::batch::to_const(a), ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(gko::batch::to_const(b), ibatch);

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
    prec.generate(A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // z = precond*r
    // rho_old = r' * z (' is for hermitian transpose)
    // p = z
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry, prec,
               z_entry, rho_old_entry, p_entry, rhs_norms_entry);

    // stopping criterion object
    StopType stop(nrhs, opts.max_its, opts.residual_tol, rhs_norms_entry.values,
                  converged);

    int iter = -1;

    while (1) {
        ++iter;

        use_implicit_norms(gko::batch::to_const(rho_old_entry), res_norms_entry,
                           converged);  // use implicit residual norms

        bool all_converged = stop.check_converged(iter, res_norms_entry.values,
                                                  {NULL, 0, 0, 0}, converged);

        logger.log_iteration(ibatch, iter, res_norms_entry.values, converged);

        if (all_converged) {
            break;
        }

        // Ap = A * p
        spmv_kernel(A_entry, gko::batch::to_const(p_entry), Ap_entry);

        // alpha = rho_old / (p' * Ap)
        // x = x + alpha * p
        // r = r - alpha * Ap
        update_x_and_r(gko::batch::to_const(rho_old_entry),
                       gko::batch::to_const(p_entry),
                       gko::batch::to_const(Ap_entry), alpha_entry, x_entry,
                       r_entry, converged);


        // z = precond * r
        prec.apply(gko::batch::to_const(r_entry), z_entry);

        // rho_new =  (r)' * (z)
        batch_dense::compute_dot_product<ValueType>(
            gko::batch::to_const(r_entry), gko::batch::to_const(z_entry),
            rho_new_entry, converged);


        // beta = rho_new / rho_old
        // p = z + beta * p
        update_p(gko::batch::to_const(rho_new_entry),
                 gko::batch::to_const(rho_old_entry),
                 gko::batch::to_const(z_entry), p_entry, converged);


        // rho_old = rho_new
        batch_dense::copy(gko::batch::to_const(rho_new_entry), rho_old_entry,
                          converged);
    }

    if (left_entry.values) {
        batch_dense::batch_scale(right_entry, x_entry);
    }
}