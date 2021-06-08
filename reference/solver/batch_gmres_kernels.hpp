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


template <typename BatchMatrixType_entry, typename PrecType, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry &A_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const PrecType &prec,
    const gko::batch_dense::BatchEntry<ValueType> &z_entry,
    const gko::batch_dense::BatchEntry<ValueType> &cs_entry,
    const gko::batch_dense::BatchEntry<ValueType> &sn_entry,
    const gko::batch_dense::BatchEntry<ValueType> &V_entry,
    const gko::batch_dense::BatchEntry<ValueType> &H_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_entry)
{
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


    // z = precond * r
    prec.apply(gko::batch::to_const(r_entry), z_entry);

    // initialize cs with zeroes
    for (int r = 0; r < cs_entry.num_rows; r++) {
        for (int c = 0; c < cs_entry.num_rhs; c++) {
            cs_entry.values[r * cs_entry.stride + c] = zero<ValueType>();
        }
    }

    // initialize sn with zeroes
    for (int r = 0; r < sn_entry.num_rows; r++) {
        for (int c = 0; c < sn_entry.num_rhs; c++) {
            sn_entry.values[r * sn_entry.stride + c] = zero<ValueType>();
        }
    }

    // initialize V with zeroes
    for (int r = 0; r < V_entry.num_rows; r++) {
        for (int c = 0; c < V_entry.num_rhs; c++) {
            V_entry.values[r * V_entry.stride + c] = zero<ValueType>();
        }
    }


    // initialize H with zeroes
    for (int r = 0; r < H_entry.num_rows; r++) {
        for (int c = 0; c < H_entry.num_rhs; c++) {
            H_entry.values[r * H_entry.stride + c] = zero<ValueType>();
        }
    }
}

template <typename ValueType>
inline void generate_plane_rotation(const ValueType *const x,
                                    const ValueType *const y, const int nrhs,
                                    ValueType *const cs, ValueType *const sn,
                                    const uint32 &converged)
{
    for (int c = 0; c < nrhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        if (y[c] == zero<ValueType>()) {
            cs[c] = one<ValueType>();
            sn[c] = zero<ValueType>();
        } else if (abs(y[c]) > abs(x[c])) {
            const ValueType temp = x[c] / y[c];
            sn[c] = (one<ValueType>() * conj(y[c])) /
                    (sqrt(one<ValueType>() + temp * temp) * y[c]);
            cs[c] = (conj(x[c]) / conj(y[c])) * sn[c];

        } else {
            const ValueType temp = y[c] / x[c];
            cs[c] = (one<ValueType>() * conj(x[c])) /
                    (sqrt(one<ValueType>() + temp * temp) * x[c]);
            sn[c] = (conj(y[c]) / conj(x[c])) * cs[c];
        }
    }
}

template <typename ValueType>
inline void apply_plane_rotation(const ValueType *const cs,
                                 const ValueType *const sn, const int nrhs,
                                 ValueType *const x, ValueType *const y,
                                 const uint32 &converged)
{
    for (int c = 0; c < nrhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        const ValueType temp = cs[c] * x[c] + sn[c] * y[c];
        y[c] = -one<ValueType>() * conj(sn[c]) * x[c] + conj(cs[c]) * y[c];
        x[c] = temp;
    }
}

template <typename ValueType>
inline void update_v_naught_and_s(
    const gko::batch_dense::BatchEntry<const ValueType> &z_entry,
    const gko::batch_dense::BatchEntry<ValueType> &V_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_entry,
    const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const gko::batch_dense::BatchEntry<ValueType> v_naught_entry{
        V_entry.values, V_entry.stride, z_entry.num_rows, V_entry.num_rhs};

    const gko::batch_dense::BatchEntry<real_type> &z_norms_entry =
        tmp_norms_entry;

    batch_dense::compute_norm2(z_entry, z_norms_entry, converged);

    for (int r = 0; r < v_naught_entry.num_rows; r++) {
        for (int c = 0; c < v_naught_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            v_naught_entry.values[r * v_naught_entry.stride + c] =
                z_entry.values[r * z_entry.stride + c] /
                static_cast<ValueType>(z_norms_entry.values[c]);
        }
    }


    for (int r = 0; r < s_entry.num_rows; r++) {
        for (int c = 0; c < s_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }
            s_entry.values[r * s_entry.stride + c] = zero<ValueType>();
        }
    }

    for (int c = 0; c < s_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        s_entry.values[c] = static_cast<ValueType>(z_norms_entry.values[c]);
    }
}


template <typename BatchMatrixEntry, typename ValueType, typename PrecType>
inline void arnoldi_method(
    const BatchMatrixEntry &A_entry, const int inner_iter,
    const gko::batch_dense::BatchEntry<ValueType> &V_entry,
    const gko::batch_dense::BatchEntry<ValueType> &H_entry,
    const gko::batch_dense::BatchEntry<ValueType> &w_entry,
    const gko::batch_dense::BatchEntry<ValueType> &helper_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_entry,
    const PrecType &prec, const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int &i = inner_iter;
    const int num_rows = w_entry.num_rows;
    const int num_rhs = w_entry.num_rhs;

    const gko::batch_dense::BatchEntry<const ValueType> v_i_entry{
        &V_entry.values[i * V_entry.stride * num_rows], V_entry.stride,
        num_rows, V_entry.num_rhs};


    spmv_kernel(A_entry, v_i_entry, helper_entry);

    prec.apply(gko::batch::to_const(helper_entry), w_entry);


    for (int k = 0; k <= i; k++) {
        const gko::batch_dense::BatchEntry<const ValueType> v_k_entry{
            &V_entry.values[k * V_entry.stride * num_rows], V_entry.stride,
            num_rows, V_entry.num_rhs};

        const gko::batch_dense::BatchEntry<ValueType> h_k_i_entry{
            &H_entry.values[k * H_entry.stride + i * num_rhs],
            static_cast<size_type>(w_entry.num_rhs), 1, w_entry.num_rhs};


        batch_dense::compute_dot_product(gko::batch::to_const(w_entry),
                                         v_k_entry, h_k_i_entry, converged);


        for (int r = 0; r < w_entry.num_rows; r++) {
            for (int c = 0; c < w_entry.num_rhs; c++) {
                const uint32 conv = converged & (1 << c);

                if (conv) {
                    continue;
                }

                ValueType h_k_i_scalar = h_k_i_entry.values[c];

                w_entry.values[r * w_entry.stride + c] -=
                    h_k_i_scalar * v_k_entry.values[r * v_k_entry.stride + c];
            }
        }
    }


    const gko::batch_dense::BatchEntry<real_type> &w_norms_entry =
        tmp_norms_entry;

    batch_dense::compute_norm2(gko::batch::to_const(w_entry), w_norms_entry,
                               converged);

    const gko::batch_dense::BatchEntry<ValueType> h_i_plus_1_i_entry{
        &H_entry.values[(i + 1) * H_entry.stride + i * num_rhs],
        static_cast<size_type>(w_entry.num_rhs), 1, w_entry.num_rhs};

    for (int c = 0; c < w_norms_entry.num_rhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        h_i_plus_1_i_entry.values[c] =
            static_cast<ValueType>(w_norms_entry.values[c]);
    }

    const gko::batch_dense::BatchEntry<ValueType> v_i_plus_1_entry{
        &V_entry.values[(i + 1) * V_entry.stride * num_rows], V_entry.stride,
        num_rows, V_entry.num_rhs};

    for (int r = 0; r < w_entry.num_rows; r++) {
        for (int c = 0; c < w_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            v_i_plus_1_entry.values[r * v_i_plus_1_entry.stride + c] =
                w_entry.values[r * w_entry.stride + c] /
                h_i_plus_1_i_entry.values[c];
        }
    }
}

template <typename ValueType>
inline void update_x(
    const int m, const gko::batch_dense::BatchEntry<const ValueType> &H_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &V_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &y_entry,
    const uint32 &converged)
{
    // upper triangular solve
    for (int r = m; r >= 0; r--) {
        for (int c = 0; c < y_entry.num_rhs; c++) {
            ValueType temp_sum = zero<ValueType>();

            for (int t = m; t > r; t--) {
                temp_sum +=
                    H_entry
                        .values[r * H_entry.stride + t * y_entry.num_rhs + c] *
                    y_entry.values[t * y_entry.stride + c];
            }

            y_entry.values[r * y_entry.stride + c] =
                (s_entry.values[r * s_entry.stride + c] - temp_sum) /
                H_entry.values[r * H_entry.stride + r * y_entry.num_rhs + c];
        }
    }


    // dense mat * vec multiplication

    for (int a = 0; a < m + 1; a++) {
        for (int r = 0; r < x_entry.num_rows; r++) {
            for (int c = 0; c < x_entry.num_rhs; c++) {
                const uint32 conv = converged & (1 << c);

                if (conv) {
                    continue;
                }

                x_entry.values[r * x_entry.stride + c] +=
                    V_entry.values[a * V_entry.stride * x_entry.num_rows +
                                   r * V_entry.stride + c] *
                    y_entry.values[a * y_entry.stride + c];
            }
        }
    }
}