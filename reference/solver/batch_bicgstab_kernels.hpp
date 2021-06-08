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


template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry &A_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_hat_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_entry,
    const gko::batch_dense::BatchEntry<ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_entry)
{
    for (int c = 0; c < rho_old_entry.num_rhs; c++) {
        rho_old_entry.values[c] = one<ValueType>();
        omega_entry.values[c] = one<ValueType>();
        alpha_entry.values[c] = one<ValueType>();
    }

    // Compute norms of rhs
    batch_dense::compute_norm2<ValueType>(b_entry, rhs_norms_entry);


    // r = b
    batch_dense::copy(b_entry, r_entry);

    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_entry,
                         gko::batch::to_const(x_entry),
                         static_cast<ValueType>(1.0), r_entry);
    batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                          res_norms_entry);

    batch_dense::copy(gko::batch::to_const(r_entry), r_hat_entry);

    for (int r = 0; r < p_entry.num_rows; r++) {
        for (int c = 0; c < p_entry.num_rhs; c++) {
            p_entry.values[r * p_entry.stride + c] = zero<ValueType>();
            v_entry.values[r * v_entry.stride + c] = zero<ValueType>();
        }
    }
}


template <typename ValueType>
inline void update_p(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &rho_old_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &p_entry,
    const uint32 &converged)
{
    for (int r = 0; r < p_entry.num_rows; r++) {
        for (int c = 0; c < p_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            const ValueType beta =
                (rho_new_entry.values[c] / rho_old_entry.values[c]) *
                (alpha_entry.values[c] / omega_entry.values[c]);

            p_entry.values[r * p_entry.stride + c] =
                r_entry.values[r * r_entry.stride + c] +
                beta * (p_entry.values[r * p_entry.stride + c] -
                        omega_entry.values[c] *
                            v_entry.values[r * v_entry.stride + c]);
        }
    }
}


template <typename ValueType>
inline void compute_alpha(
    const gko::batch_dense::BatchEntry<const ValueType> &rho_new_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const uint32 &converged)
{
    batch_dense::compute_dot_product<ValueType>(r_hat_entry, v_entry,
                                                alpha_entry, converged);

    for (int c = 0; c < alpha_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        alpha_entry.values[c] = rho_new_entry.values[c] / alpha_entry.values[c];
    }
}


template <typename ValueType>
inline void update_s(
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_entry,
    const uint32 &converged)
{
    for (int r = 0; r < s_entry.num_rows; r++) {
        for (int c = 0; c < s_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }
            s_entry.values[r * s_entry.stride + c] =
                r_entry.values[r * r_entry.stride + c] -
                alpha_entry.values[c] * v_entry.values[r * v_entry.stride + c];
        }
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<ValueType> &temp_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_entry,
    const uint32 &converged)
{
    batch_dense::compute_dot_product<ValueType>(t_entry, s_entry, omega_entry,
                                                converged);
    batch_dense::compute_dot_product<ValueType>(t_entry, t_entry, temp_entry,
                                                converged);

    for (int c = 0; c < omega_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }
        omega_entry.values[c] /= temp_entry.values[c];
    }
}


template <typename ValueType>
inline void update_x_and_r(
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_hat_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const uint32 &converged)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        for (int c = 0; c < x_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            const ValueType omega = omega_entry.values[c];

            x_entry.values[r * x_entry.stride + c] =
                x_entry.values[r * x_entry.stride + c] +
                alpha_entry.values[c] *
                    p_hat_entry.values[r * p_hat_entry.stride + c] +
                omega * s_hat_entry.values[r * s_hat_entry.stride + c];

            r_entry.values[r * r_entry.stride + c] =
                s_entry.values[r * s_entry.stride + c] -
                omega * t_entry.values[r * t_entry.stride + c];
        }
    }
}

template <typename ValueType>
inline void update_x_middle(
    const gko::batch_dense::BatchEntry<const ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &p_hat_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const uint32 &converged_recent)
{
    for (int r = 0; r < x_entry.num_rows; r++) {
        for (int c = 0; c < x_entry.num_rhs; c++) {
            const uint32 conv = converged_recent & (1 << c);

            if (conv) {
                x_entry.values[r * x_entry.stride + c] =
                    x_entry.values[r * x_entry.stride + c] +
                    alpha_entry.values[c] *
                        p_hat_entry.values[r * p_hat_entry.stride + c];
            }
        }
    }
}
