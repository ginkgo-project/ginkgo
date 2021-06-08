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


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType(dist(gen), dist(gen));
}

template <typename ValueType>
inline void orthonormalize_subspace_vectors(
    const gko::batch_dense::BatchEntry<ValueType> &Subspace_vectors_entry,
    const int num_rows, const int subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType> &temp_for_single_rhs_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    for (int i = 0; i < subspace_dim; i++) {
        const gko::batch_dense::BatchEntry<ValueType> p_i_entry{
            &Subspace_vectors_entry
                 .values[i * num_rows * Subspace_vectors_entry.stride],
            Subspace_vectors_entry.stride, num_rows, 1};

        const gko::batch_dense::BatchEntry<ValueType> &w_i_entry =
            temp_for_single_rhs_entry;

        // w_i = p_i
        batch_dense::copy(gko::batch::to_const(p_i_entry), w_i_entry);

        for (int j = 0; j < i; j++) {
            // w_i = w_i - proj(p_i) on w_j that is w_i = w_i - (< w_j , p_i >
            // /< w_j , w_j > ) * w_j

            const gko::batch_dense::BatchEntry<ValueType> w_j_entry{
                &Subspace_vectors_entry
                     .values[j * num_rows * Subspace_vectors_entry.stride],
                Subspace_vectors_entry.stride, num_rows, 1};

            ValueType mul;
            const gko::batch_dense::BatchEntry<ValueType> mul_entry{&mul, 1, 1,
                                                                    1};
            batch_dense::compute_dot_product(gko::batch::to_const(w_j_entry),
                                             gko::batch::to_const(p_i_entry),
                                             mul_entry);
            mul_entry.values[0] /= static_cast<ValueType>(
                tmp_norms_entry.values[j] * tmp_norms_entry.values[j]);

            mul_entry.values[0] *= -one<ValueType>();
            batch_dense::add_scaled(gko::batch::to_const(mul_entry),
                                    gko::batch::to_const(w_j_entry), w_i_entry);
        }

        // p_i = w_i
        batch_dense::copy(gko::batch::to_const(w_i_entry), p_i_entry);

        batch_dense::compute_norm2(gko::batch::to_const(w_i_entry),
                                   gko::batch_dense::BatchEntry<real_type>{
                                       &tmp_norms_entry.values[i], 1, 1, 1});
    }

    // e_k = w_k / || w_k ||
    for (int k = 0; k < subspace_dim; k++) {
        const gko::batch_dense::BatchEntry<ValueType> w_k_entry{
            &Subspace_vectors_entry
                 .values[k * num_rows * Subspace_vectors_entry.stride],
            Subspace_vectors_entry.stride, num_rows, 1};

        ValueType scale_factor =
            one<ValueType>() /
            static_cast<ValueType>(tmp_norms_entry.values[k]);

        batch_dense::scale(
            gko::batch_dense::BatchEntry<const ValueType>{&scale_factor, 1, 1,
                                                          1},
            w_k_entry);
    }
}

template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry &A_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &b_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &G_entry,
    const gko::batch_dense::BatchEntry<ValueType> &U_entry,
    const gko::batch_dense::BatchEntry<ValueType> &M_entry,
    const gko::batch_dense::BatchEntry<ValueType> &Subspace_vectors_entry,
    const bool deterministic,
    const gko::batch_dense::BatchEntry<ValueType> &xs_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rs_entry,
    const bool smoothing,
    const gko::batch_dense::BatchEntry<ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &res_norms_entry,
    const gko::batch_dense::BatchEntry<ValueType> &temp_for_single_rhs_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>
        &tmp_norms_entry)
{
    // Compute norms of rhs
    batch_dense::compute_norm2<ValueType>(b_entry, rhs_norms_entry);

    const auto subspace_dim = M_entry.num_rows;
    const auto num_rows = b_entry.num_rows;
    const auto num_rhs = b_entry.num_rhs;

    // r = b
    batch_dense::copy(b_entry, r_entry);
    // r = b - A*x
    advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_entry,
                         gko::batch::to_const(x_entry),
                         static_cast<ValueType>(1.0), r_entry);
    // compute residual norms
    batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                          res_norms_entry);

    // omega = 1
    for (int c = 0; c < omega_entry.num_rhs; c++) {
        omega_entry.values[c] = one<ValueType>();
    }

    if (smoothing == true) {
        batch_dense::copy(x_entry, xs_entry);
        batch_dense::copy(gko::batch::to_const(r_entry), rs_entry);
    }

    // initialize G,U with zeroes
    for (int vec_index = 0; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < num_rows; row_index++) {
            for (int rhs_index = 0; rhs_index < num_rhs; rhs_index++) {
                G_entry.values[vec_index * num_rows * G_entry.stride +
                               row_index * G_entry.stride + rhs_index] =
                    zero<ValueType>();

                U_entry.values[vec_index * num_rows * U_entry.stride +
                               row_index * U_entry.stride + rhs_index] =
                    zero<ValueType>();
            }
        }
    }

    // M = identity
    for (int row_index = 0; row_index < subspace_dim; row_index++) {
        for (int col_index = 0; col_index < subspace_dim; col_index++) {
            for (int rhs_index = 0; rhs_index < num_rhs; rhs_index++) {
                ValueType val = zero<ValueType>();

                if (row_index == col_index) {
                    val = one<ValueType>();
                }

                M_entry.values[row_index * M_entry.stride +
                               col_index * num_rhs + rhs_index] = val;
            }
        }
    }

    auto dist = std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);
    auto seed = deterministic ? 15 : time(NULL);
    auto gen = std::ranlux48(seed);

    // initialize Subspace_vectors
    for (int vec_index = 0; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < num_rows; row_index++) {
            ValueType val = get_rand_value<ValueType>(dist, gen);

            Subspace_vectors_entry
                .values[vec_index * Subspace_vectors_entry.stride * num_rows +
                        row_index * Subspace_vectors_entry.stride] = val;
        }
    }

    // orthonormailize Subspace_vectors
    orthonormalize_subspace_vectors(Subspace_vectors_entry, num_rows,
                                    subspace_dim, temp_for_single_rhs_entry,
                                    tmp_norms_entry);
}


template <typename ValueType>
inline void update_f(
    const gko::batch_dense::BatchEntry<const ValueType> &Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const size_type subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType> &f_entry,
    const uint32 &converged)
{
    for (int vec_index = 0; vec_index < subspace_dim; vec_index++) {
        for (int rhs_index = 0; rhs_index < r_entry.num_rhs; rhs_index++) {
            const uint32 conv = converged & (1 << rhs_index);

            if (conv) {
                continue;
            }

            f_entry.values[vec_index * f_entry.stride + rhs_index] =
                zero<ValueType>();
        }

        for (int row_index = 0; row_index < r_entry.num_rows; row_index++) {
            ValueType P_val =
                Subspace_vectors_entry
                    .values[vec_index * Subspace_vectors_entry.stride *
                                r_entry.num_rows +
                            row_index * Subspace_vectors_entry.stride];

            for (int rhs_index = 0; rhs_index < r_entry.num_rhs; rhs_index++) {
                const uint32 conv = converged & (1 << rhs_index);

                if (conv) {
                    continue;
                }

                f_entry.values[vec_index * f_entry.stride + rhs_index] +=
                    conj(P_val) *
                    r_entry.values[row_index * r_entry.stride + rhs_index];
            }
        }
    }
}

template <typename ValueType>
inline void update_c(
    const gko::batch_dense::BatchEntry<const ValueType> &M_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &f_entry,
    const gko::batch_dense::BatchEntry<ValueType> &c_entry,
    const gko::batch_dense::BatchEntry<ValueType> &temp_sum_entry,
    const uint32 &converged)
{
    const auto subspace_dim = M_entry.num_rows;
    const auto nrhs = f_entry.num_rhs;
    // upper triangular solve
    // solve top to bottom
    for (int row_index = 0; row_index < subspace_dim; row_index++) {
        for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
            temp_sum_entry.values[rhs_index] = zero<ValueType>();
        }

        for (int col_index = 0; col_index < row_index; col_index++) {
            for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
                const uint32 conv = converged & (1 << rhs_index);

                if (conv) {
                    continue;
                }
                temp_sum_entry.values[rhs_index] +=
                    M_entry.values[row_index * M_entry.stride +
                                   col_index * nrhs + rhs_index] *
                    c_entry.values[col_index * c_entry.stride + rhs_index];
            }
        }

        for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
            const uint32 conv = converged & (1 << rhs_index);

            if (conv) {
                continue;
            }
            c_entry.values[row_index * c_entry.stride + rhs_index] =
                (f_entry.values[row_index * f_entry.stride + rhs_index] -
                 temp_sum_entry.values[rhs_index]) /
                M_entry.values[row_index * M_entry.stride + row_index * nrhs +
                               rhs_index];
        }
    }
}

template <typename ValueType>
inline void update_v(
    const gko::batch_dense::BatchEntry<const ValueType> &G_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &c_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &v_entry, const size_type k,
    const uint32 &converged)
{
    batch_dense::copy(gko::batch::to_const(r_entry), v_entry, converged);

    const auto subspace_dim = c_entry.num_rows;
    const auto nrows = r_entry.num_rows;

    for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < v_entry.num_rows; row_index++) {
            for (int rhs_index = 0; rhs_index < v_entry.num_rhs; rhs_index++) {
                const uint32 conv = converged & (1 << rhs_index);

                if (conv) {
                    continue;
                }

                v_entry.values[row_index * v_entry.stride + rhs_index] -=
                    G_entry.values[vec_index * nrows * G_entry.stride +
                                   row_index * G_entry.stride + rhs_index] *
                    c_entry.values[vec_index * c_entry.stride + rhs_index];
            }
        }
    }
}

template <typename ValueType>
inline void update_u_k(
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &c_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_entry,
    const size_type k,
    const gko::batch_dense::BatchEntry<ValueType> &helper_entry,
    const gko::batch_dense::BatchEntry<ValueType> &u_k_entry,
    const uint32 &converged)
{
    const auto subspace_dim = c_entry.num_rows;
    const auto nrows = v_entry.num_rows;
    const auto nrhs = v_entry.num_rhs;

    for (int row_index = 0; row_index < nrows; row_index++) {
        for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
            helper_entry.values[row_index * helper_entry.stride + rhs_index] =
                omega_entry.values[rhs_index] *
                v_entry.values[row_index * v_entry.stride + rhs_index];
        }
    }

    for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < nrows; row_index++) {
            for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
                helper_entry
                    .values[row_index * helper_entry.stride + rhs_index] +=
                    U_entry.values[vec_index * nrows * U_entry.stride +
                                   row_index * U_entry.stride + rhs_index] *
                    c_entry.values[vec_index * c_entry.stride + rhs_index];
            }
        }
    }

    batch_dense::copy(gko::batch::to_const(helper_entry), u_k_entry, converged);
}


template <typename ValueType>
inline void update_g_k_and_u_k(
    const size_type k,
    const gko::batch_dense::BatchEntry<const ValueType> &G_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &M_entry,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType> &g_k_entry,
    const gko::batch_dense::BatchEntry<ValueType> &u_k_entry,
    const uint32 &converged)
{
    const auto nrows = g_k_entry.num_rows;
    const auto nrhs = g_k_entry.num_rhs;
    const auto subspace_dim = M_entry.num_rows;


    for (int i = 0; i <= static_cast<int>(k) - 1; i++) {
        // alpha = (p_i * g_k)/M(i,i)
        for (int rhs = 0; rhs < nrhs; rhs++) {
            alpha_entry.values[rhs] = zero<ValueType>();
        }

        for (int row = 0; row < nrows; row++) {
            ValueType p_val =
                Subspace_vectors_entry
                    .values[i * nrows * Subspace_vectors_entry.stride +
                            row * Subspace_vectors_entry.stride];
            for (int rhs = 0; rhs < nrhs; rhs++) {
                alpha_entry.values[rhs] +=
                    (conj(p_val) *
                     g_k_entry.values[row * g_k_entry.stride + rhs]) /
                    M_entry.values[i * M_entry.stride + i * nrhs + rhs];
            }
        }


        // g_k = g_k - alpha * g_i
        // u_k = u_k - alpha * u_i
        for (int row = 0; row < nrows; row++) {
            for (int rhs = 0; rhs < nrhs; rhs++) {
                const uint32 conv = converged & (1 << rhs);

                if (conv) {
                    continue;
                }

                const ValueType alpha = alpha_entry.values[rhs];

                g_k_entry.values[row * g_k_entry.stride + rhs] -=
                    alpha * G_entry.values[i * nrows * G_entry.stride +
                                           row * G_entry.stride + rhs];

                u_k_entry.values[row * u_k_entry.stride + rhs] -=
                    alpha * U_entry.values[i * nrows * U_entry.stride +
                                           row * U_entry.stride + rhs];
            }
        }
    }
}


template <typename ValueType>
inline void update_M(
    const gko::batch_dense::BatchEntry<const ValueType> &g_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<ValueType> &M_entry, const size_type k,
    const uint32 &converged)
{
    const auto subspace_dim = M_entry.num_rows;
    const auto nrows = g_k_entry.num_rows;
    const auto nrhs = g_k_entry.num_rhs;

    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
    for (int i = k; i < subspace_dim; i++) {
        for (int rhs = 0; rhs < nrhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            M_entry.values[i * M_entry.stride + k * nrhs + rhs] =
                zero<ValueType>();
        }

        for (int row = 0; row < nrows; row++) {
            ValueType p_val =
                Subspace_vectors_entry
                    .values[i * nrows * Subspace_vectors_entry.stride +
                            row * Subspace_vectors_entry.stride];
            for (int rhs = 0; rhs < nrhs; rhs++) {
                const uint32 conv = converged & (1 << rhs);

                if (conv) {
                    continue;
                }

                M_entry.values[i * M_entry.stride + k * nrhs + rhs] +=
                    conj(p_val) *
                    g_k_entry.values[row * g_k_entry.stride + rhs];
            }
        }
    }
}


template <typename ValueType>
inline void update_r_and_x_inner_loop(
    const gko::batch_dense::BatchEntry<const ValueType> &g_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &u_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const uint32 &converged)
{
    for (int row = 0; row < g_k_entry.num_rows; row++) {
        for (int rhs = 0; rhs < g_k_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            const ValueType beta = beta_entry.values[rhs];

            r_entry.values[row * r_entry.stride + rhs] -=
                beta * g_k_entry.values[row * g_k_entry.stride + rhs];


            x_entry.values[row * x_entry.stride + rhs] +=
                beta * u_k_entry.values[row * u_k_entry.stride + rhs];
        }
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_entry,
    const gko::batch_dense::BatchEntry<ValueType> &t_r_dot_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_t_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &omega_entry,
    const gko::remove_complex<ValueType> kappa, const uint32 &converged)
{
    batch_dense::compute_dot_product(gko::batch::to_const(t_entry),
                                     gko::batch::to_const(r_entry),
                                     t_r_dot_entry, converged);

    batch_dense::compute_norm2(gko::batch::to_const(t_entry), norms_t_entry,
                               converged);


    batch_dense::compute_norm2(gko::batch::to_const(r_entry), norms_r_entry,
                               converged);

    // omega = ( t * r )/ (t * t)
    for (int rhs = 0; rhs < omega_entry.num_rhs; rhs++) {
        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        omega_entry.values[rhs] =
            t_r_dot_entry.values[rhs] /
            static_cast<ValueType>(norms_t_entry.values[rhs] *
                                   norms_t_entry.values[rhs]);
    }

    // rho = (t * r ) /(||t|| * || r||)
    for (int rhs = 0; rhs < rho_entry.num_rhs; rhs++) {
        rho_entry.values[rhs] =
            t_r_dot_entry.values[rhs] /
            static_cast<ValueType>(norms_t_entry.values[rhs] *
                                   norms_r_entry.values[rhs]);
    }

    // if |rho| < kappa
    //      omega = omega * kappa / |rho|
    // end if
    for (int rhs = 0; rhs < rho_entry.num_rhs; rhs++) {
        const uint32 conv = converged & (1 << rhs);

        if (conv) {
            continue;
        }

        if (abs(rho_entry.values[rhs]) < kappa) {
            omega_entry.values[rhs] *= kappa / abs(rho_entry.values[rhs]);
        }
    }
}


template <typename ValueType>
inline void update_r_and_x_outer_loop(
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const uint32 &converged)
{
    for (int row = 0; row < t_entry.num_rows; row++) {
        for (int rhs = 0; rhs < t_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            const ValueType omega = omega_entry.values[rhs];

            r_entry.values[row * r_entry.stride + rhs] -=
                omega * t_entry.values[row * t_entry.stride + rhs];

            x_entry.values[row * x_entry.stride + rhs] +=
                omega * v_entry.values[row * v_entry.stride + rhs];
        }
    }
}

template <typename ValueType>
inline void smoothing_operation(
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &gamma_entry,
    const gko::batch_dense::BatchEntry<ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<ValueType> &xs_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rs_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_t_entry,
    const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;

    // t = rs - r
    for (int row = 0; row < t_entry.num_rows; row++) {
        for (int rhs = 0; rhs < t_entry.num_rhs; rhs++) {
            t_entry.values[row * t_entry.stride + rhs] =
                rs_entry.values[row * rs_entry.stride + rhs] -
                r_entry.values[row * r_entry.stride + rhs];
        }
    }


    // gamma = (t * rs)/(t * t)
    batch_dense::compute_dot_product(gko::batch::to_const(t_entry),
                                     gko::batch::to_const(rs_entry),
                                     gamma_entry, converged);
    batch_dense::compute_norm2(gko::batch::to_const(t_entry), norms_t_entry,
                               converged);

    for (int rhs = 0; rhs < t_entry.num_rhs; rhs++) {
        gamma_entry.values[rhs] /= static_cast<ValueType>(
            norms_t_entry.values[rhs] * norms_t_entry.values[rhs]);
    }


    // rs = rs - gamma*(rs - r)
    // xs = xs - gamma*(xs - x)
    for (int row = 0; row < rs_entry.num_rows; row++) {
        for (int rhs = 0; rhs < rs_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            rs_entry.values[row * rs_entry.stride + rhs] =
                (one<ValueType>() - gamma_entry.values[rhs]) *
                    rs_entry.values[row * rs_entry.stride + rhs] +
                gamma_entry.values[rhs] *
                    r_entry.values[row * r_entry.stride + rhs];

            xs_entry.values[row * xs_entry.stride + rhs] =
                (one<ValueType>() - gamma_entry.values[rhs]) *
                    xs_entry.values[row * xs_entry.stride + rhs] +
                gamma_entry.values[rhs] *
                    x_entry.values[row * x_entry.stride + rhs];
        }
    }
}
