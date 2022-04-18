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

template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return ValueType(dist(gen), dist(gen));
}

template <typename ValueType>
inline void orthonormalize_subspace_vectors(
    const gko::batch_dense::BatchEntry<ValueType>& Subspace_vectors_entry,
    const int num_rows, const int subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType>& temp_for_single_rhs_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        tmp_norms_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    for (int i = 0; i < subspace_dim; i++) {
        const gko::batch_dense::BatchEntry<ValueType> p_i_entry{
            &Subspace_vectors_entry
                 .values[i * num_rows * Subspace_vectors_entry.stride],
            Subspace_vectors_entry.stride, num_rows, 1};

        const gko::batch_dense::BatchEntry<ValueType>& w_i_entry =
            temp_for_single_rhs_entry;

        // w_i = p_i
        copy(gko::batch::to_const(p_i_entry), w_i_entry);

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
            compute_dot_product(gko::batch::to_const(w_j_entry),
                                gko::batch::to_const(p_i_entry), mul_entry);
            mul_entry.values[0] /= static_cast<ValueType>(
                tmp_norms_entry.values[j] * tmp_norms_entry.values[j]);

            mul_entry.values[0] *= -one<ValueType>();
            add_scaled(gko::batch::to_const(mul_entry),
                       gko::batch::to_const(w_j_entry), w_i_entry);
        }

        // p_i = w_i
        copy(gko::batch::to_const(w_i_entry), p_i_entry);

        compute_norm2(gko::batch::to_const(w_i_entry),
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

        scale(gko::batch_dense::BatchEntry<const ValueType>{&scale_factor, 1, 1,
                                                            1},
              w_k_entry);
    }
}

template <typename BatchMatrixType_entry, typename ValueType>
inline void initialize(
    const BatchMatrixType_entry& A_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& b_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& G_entry,
    const gko::batch_dense::BatchEntry<ValueType>& U_entry,
    const gko::batch_dense::BatchEntry<ValueType>& M_entry,
    const gko::batch_dense::BatchEntry<ValueType>& Subspace_vectors_entry,
    const bool deterministic,
    const gko::batch_dense::BatchEntry<ValueType>& xs_entry,
    const gko::batch_dense::BatchEntry<ValueType>& rs_entry,
    const bool smoothing,
    const gko::batch_dense::BatchEntry<ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        rhs_norms_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        res_norms_entry,
    const gko::batch_dense::BatchEntry<ValueType>& temp_for_single_rhs_entry,
    const gko::batch_dense::BatchEntry<typename gko::remove_complex<ValueType>>&
        tmp_norms_entry)
{
    // Compute norms of rhs
    compute_norm2<ValueType>(b_entry, rhs_norms_entry);

    const auto subspace_dim = M_entry.num_rows;
    const auto num_rows = b_entry.num_rows;
    const auto num_rhs = b_entry.num_rhs;

    // r = b
    copy(b_entry, r_entry);
    // r = b - A*x
    advanced_matvec_kernel(static_cast<ValueType>(-1.0), A_entry,
                           gko::batch::to_const(x_entry),
                           static_cast<ValueType>(1.0), r_entry);
    // compute residual norms
    compute_norm2<ValueType>(gko::batch::to_const(r_entry), res_norms_entry);

    // omega = 1
    for (int c = 0; c < omega_entry.num_rhs; c++) {
        omega_entry.values[c] = one<ValueType>();
    }

    if (smoothing == true) {
        copy(x_entry, xs_entry);
        copy(gko::batch::to_const(r_entry), rs_entry);
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
            ValueType val = zero<ValueType>();
            if (row_index == col_index) {
                val = one<ValueType>();
            }
            M_entry.values[row_index * M_entry.stride + col_index] = val;
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
    const gko::batch_dense::BatchEntry<const ValueType>& Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const size_type subspace_dim,
    const gko::batch_dense::BatchEntry<ValueType>& f_entry)
{
    for (int vec_index = 0; vec_index < subspace_dim; vec_index++) {
        f_entry.values[vec_index * f_entry.stride] = zero<ValueType>();

        for (int row_index = 0; row_index < r_entry.num_rows; row_index++) {
            ValueType P_val =
                Subspace_vectors_entry
                    .values[vec_index * Subspace_vectors_entry.stride *
                                r_entry.num_rows +
                            row_index * Subspace_vectors_entry.stride];
            f_entry.values[vec_index * f_entry.stride] +=
                conj(P_val) * r_entry.values[row_index * r_entry.stride];
        }
    }
}

template <typename ValueType>
inline void update_c(
    const gko::batch_dense::BatchEntry<const ValueType>& M_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& f_entry,
    const gko::batch_dense::BatchEntry<ValueType>& c_entry,
    const gko::batch_dense::BatchEntry<ValueType>& temp_sum_entry)
{
    const auto subspace_dim = M_entry.num_rows;
    // upper triangular solve
    // solve top to bottom
    for (int row_index = 0; row_index < subspace_dim; row_index++) {
        temp_sum_entry.values[0] = zero<ValueType>();

        for (int col_index = 0; col_index < row_index; col_index++) {
            temp_sum_entry.values[0] +=
                M_entry.values[row_index * M_entry.stride + col_index] *
                c_entry.values[col_index * c_entry.stride];
        }

        c_entry.values[row_index * c_entry.stride] =
            (f_entry.values[row_index * f_entry.stride] -
             temp_sum_entry.values[0]) /
            M_entry.values[row_index * M_entry.stride + row_index];
    }
}

template <typename ValueType>
inline void update_v(
    const gko::batch_dense::BatchEntry<const ValueType>& G_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& c_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& v_entry, const size_type k)
{
    copy(gko::batch::to_const(r_entry), v_entry);

    const auto subspace_dim = c_entry.num_rows;
    const auto nrows = r_entry.num_rows;

    for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < v_entry.num_rows; row_index++) {
            v_entry.values[row_index * v_entry.stride] -=
                G_entry.values[vec_index * nrows * G_entry.stride +
                               row_index * G_entry.stride] *
                c_entry.values[vec_index * c_entry.stride];
        }
    }
}

template <typename ValueType>
inline void update_u_k(
    const gko::batch_dense::BatchEntry<const ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& c_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& U_entry,
    const size_type k,
    const gko::batch_dense::BatchEntry<ValueType>& helper_entry,
    const gko::batch_dense::BatchEntry<ValueType>& u_k_entry)
{
    const auto subspace_dim = c_entry.num_rows;
    const auto nrows = v_entry.num_rows;

    for (int row_index = 0; row_index < nrows; row_index++) {
        helper_entry.values[row_index * helper_entry.stride] =
            omega_entry.values[0] * v_entry.values[row_index * v_entry.stride];
    }

    for (int vec_index = k; vec_index < subspace_dim; vec_index++) {
        for (int row_index = 0; row_index < nrows; row_index++) {
            helper_entry.values[row_index * helper_entry.stride] +=
                U_entry.values[vec_index * nrows * U_entry.stride +
                               row_index * U_entry.stride] *
                c_entry.values[vec_index * c_entry.stride];
        }
    }

    copy(gko::batch::to_const(helper_entry), u_k_entry);
}


template <typename ValueType>
inline void update_g_k_and_u_k(
    const size_type k,
    const gko::batch_dense::BatchEntry<const ValueType>& G_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& U_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& M_entry,
    const gko::batch_dense::BatchEntry<ValueType>& alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType>& g_k_entry,
    const gko::batch_dense::BatchEntry<ValueType>& u_k_entry)
{
    const auto nrows = g_k_entry.num_rows;
    for (int i = 0; i <= static_cast<int>(k) - 1; i++) {
        // alpha = (p_i * g_k)/M(i,i)
        alpha_entry.values[0] = zero<ValueType>();
        for (int row = 0; row < nrows; row++) {
            ValueType p_val =
                Subspace_vectors_entry
                    .values[i * nrows * Subspace_vectors_entry.stride +
                            row * Subspace_vectors_entry.stride];
            alpha_entry.values[0] +=
                (conj(p_val) * g_k_entry.values[row * g_k_entry.stride]) /
                M_entry.values[i * M_entry.stride + i];
        }

        // g_k = g_k - alpha * g_i
        // u_k = u_k - alpha * u_i
        for (int row = 0; row < nrows; row++) {
            const ValueType alpha = alpha_entry.values[0];
            g_k_entry.values[row * g_k_entry.stride] -=
                alpha *
                G_entry
                    .values[i * nrows * G_entry.stride + row * G_entry.stride];
            u_k_entry.values[row * u_k_entry.stride] -=
                alpha *
                U_entry
                    .values[i * nrows * U_entry.stride + row * U_entry.stride];
        }
    }
}


template <typename ValueType>
inline void update_M(
    const gko::batch_dense::BatchEntry<const ValueType>& g_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<ValueType>& M_entry, const size_type k)
{
    const auto subspace_dim = M_entry.num_rows;
    const auto nrows = g_k_entry.num_rows;
    // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
    for (int i = k; i < subspace_dim; i++) {
        M_entry.values[i * M_entry.stride + k] = zero<ValueType>();

        for (int row = 0; row < nrows; row++) {
            ValueType p_val =
                Subspace_vectors_entry
                    .values[i * nrows * Subspace_vectors_entry.stride +
                            row * Subspace_vectors_entry.stride];
            M_entry.values[i * M_entry.stride + k] +=
                conj(p_val) * g_k_entry.values[row * g_k_entry.stride];
        }
    }
}


template <typename ValueType>
inline void update_r_and_x_inner_loop(
    const gko::batch_dense::BatchEntry<const ValueType>& g_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& u_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& beta_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    for (int row = 0; row < g_k_entry.num_rows; row++) {
        const ValueType beta = beta_entry.values[0];
        r_entry.values[row * r_entry.stride] -=
            beta * g_k_entry.values[row * g_k_entry.stride];
        x_entry.values[row * x_entry.stride] +=
            beta * u_k_entry.values[row * u_k_entry.stride];
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch_dense::BatchEntry<const ValueType>& t_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& rho_entry,
    const gko::batch_dense::BatchEntry<ValueType>& t_r_dot_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>&
        norms_t_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>&
        norms_r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& omega_entry,
    const gko::remove_complex<ValueType> kappa)
{
    compute_dot_product(gko::batch::to_const(t_entry),
                        gko::batch::to_const(r_entry), t_r_dot_entry);
    compute_norm2(gko::batch::to_const(t_entry), norms_t_entry);
    compute_norm2(gko::batch::to_const(r_entry), norms_r_entry);

    // omega = ( t * r )/ (t * t)
    omega_entry.values[0] = t_r_dot_entry.values[0] /
                            static_cast<ValueType>(norms_t_entry.values[0] *
                                                   norms_t_entry.values[0]);

    // rho = (t * r ) /(||t|| * || r||)
    rho_entry.values[0] = t_r_dot_entry.values[0] /
                          static_cast<ValueType>(norms_t_entry.values[0] *
                                                 norms_r_entry.values[0]);

    // if |rho| < kappa
    //      omega = omega * kappa / |rho|
    // end if
    if (abs(rho_entry.values[0]) < kappa) {
        omega_entry.values[0] *= kappa / abs(rho_entry.values[0]);
    }
}


template <typename ValueType>
inline void update_r_and_x_outer_loop(
    const gko::batch_dense::BatchEntry<const ValueType>& t_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& v_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& omega_entry,
    const gko::batch_dense::BatchEntry<ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& x_entry)
{
    for (int row = 0; row < t_entry.num_rows; row++) {
        const ValueType omega = omega_entry.values[0];
        r_entry.values[row * r_entry.stride] -=
            omega * t_entry.values[row * t_entry.stride];

        x_entry.values[row * x_entry.stride] +=
            omega * v_entry.values[row * v_entry.stride];
    }
}

template <typename ValueType>
inline void smoothing_operation(
    const gko::batch_dense::BatchEntry<const ValueType>& x_entry,
    const gko::batch_dense::BatchEntry<const ValueType>& r_entry,
    const gko::batch_dense::BatchEntry<ValueType>& gamma_entry,
    const gko::batch_dense::BatchEntry<ValueType>& t_entry,
    const gko::batch_dense::BatchEntry<ValueType>& xs_entry,
    const gko::batch_dense::BatchEntry<ValueType>& rs_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>&
        norms_t_entry)
{
    using real_type = typename gko::remove_complex<ValueType>;

    // t = rs - r
    for (int row = 0; row < t_entry.num_rows; row++) {
        t_entry.values[row * t_entry.stride] =
            rs_entry.values[row * rs_entry.stride] -
            r_entry.values[row * r_entry.stride];
    }

    // gamma = (t * rs)/(t * t)
    compute_dot_product(gko::batch::to_const(t_entry),
                        gko::batch::to_const(rs_entry), gamma_entry);
    compute_norm2(gko::batch::to_const(t_entry), norms_t_entry);

    gamma_entry.values[0] /= static_cast<ValueType>(norms_t_entry.values[0] *
                                                    norms_t_entry.values[0]);

    // rs = rs - gamma*(rs - r)
    // xs = xs - gamma*(xs - x)
    for (int row = 0; row < rs_entry.num_rows; row++) {
        rs_entry.values[row * rs_entry.stride] =
            (one<ValueType>() - gamma_entry.values[0]) *
                rs_entry.values[row * rs_entry.stride] +
            gamma_entry.values[0] * r_entry.values[row * r_entry.stride];

        xs_entry.values[row * xs_entry.stride] =
            (one<ValueType>() - gamma_entry.values[0]) *
                xs_entry.values[row * xs_entry.stride] +
            gamma_entry.values[0] * x_entry.values[row * x_entry.stride];
    }
}


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
inline void batch_entry_idr_impl(
    const gko::kernels::batch_idr::BatchIdrOptions<remove_complex<ValueType>>&
        opts,
    LogType logger, PrecType prec, const BatchMatrixType& a,
    const gko::batch_dense::UniformBatch<const ValueType>& b,
    const gko::batch_dense::UniformBatch<ValueType>& x, const size_type ibatch,
    unsigned char* const local_space)
{
    using real_type = typename gko::remove_complex<ValueType>;
    constexpr int max_nrhs = max_num_rhs;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    const auto subspace_dim = opts.subspace_dim_val;
    const auto kappa = opts.kappa_val;
    const auto smoothing = opts.to_use_smoothing;
    const auto deterministic = opts.deterministic_gen;

    ValueType* const r = reinterpret_cast<ValueType*>(local_space);
    ValueType* const t = r + nrows * nrhs;
    ValueType* const v = t + nrows * nrhs;
    ValueType* const xs = v + nrows * nrhs;
    ValueType* const rs = xs + nrows * nrhs;
    ValueType* const helper = rs + nrows * nrhs;
    ValueType* const f = helper + nrows * nrhs;
    ValueType* const c = f + subspace_dim * nrhs;
    ValueType* const Subspace_vectors = c + subspace_dim * nrhs;
    ValueType* const G = Subspace_vectors + nrows * subspace_dim;
    ValueType* const U = G + nrows * subspace_dim * nrhs;
    ValueType* const M = U + nrows * subspace_dim * nrhs;
    ValueType* const prec_work = M + subspace_dim * subspace_dim * nrhs;
    ValueType* const temp_for_single_rhs =
        prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz);
    real_type* const norms_tmp =
        reinterpret_cast<real_type*>(temp_for_single_rhs + nrows * nrhs);
    ValueType omega[max_nrhs];
    ValueType temp1[max_nrhs];
    ValueType temp2[max_nrhs];
    real_type norms_t[max_nrhs];
    real_type norms_r[max_nrhs];
    real_type norms_rhs[max_nrhs];
    real_type norms_res[max_nrhs];

    // uint32 converged = 0;

    // const typename BatchMatrixType::entry_type A_entry =
    const auto A_entry = gko::batch::batch_entry(a, ibatch);

    const gko::batch_dense::BatchEntry<const ValueType> b_entry =
        gko::batch::batch_entry(b, ibatch);

    const gko::batch_dense::BatchEntry<ValueType> x_entry =
        gko::batch::batch_entry(x, ibatch);

    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix
    const gko::batch_dense::BatchEntry<ValueType> r_entry{
        r, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> t_entry{
        t, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> v_entry{
        v, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> helper_entry{
        helper, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> xs_entry{
        xs, static_cast<size_type>(nrhs), nrows, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> rs_entry{
        rs, static_cast<size_type>(nrhs), nrows, nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix
    const gko::batch_dense::BatchEntry<ValueType> f_entry{
        f, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim), nrhs};
    const gko::batch_dense::BatchEntry<ValueType> c_entry{
        c, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim), nrhs};
    // storage:row-major , residual vector corresponding to each rhs is
    // stored as a col. of the matrix

    // P = [ p_0 , p_1 , ... , p_(subspace_dim - 1) ] , subspace S is the
    // left null space of matrix P to store subspace defining vectors: p_i ,
    // i = 0, ..., subspace_dim -1 , we use a matrix named Subspace_vectors
    const gko::batch_dense::BatchEntry<ValueType> Subspace_vectors_entry{
        Subspace_vectors, 1, nrows * static_cast<int>(subspace_dim), 1};
    // storage:row-major order , subspace vectors
    // are stored in a single col. one after the other-(matrix
    // Subspace_vectors on paper). And to get p_i : that is ith subspace
    // vector : p_i_entry{  &Subspace_vectors[i*
    // Subspace_vectors_entry.stride * nrows], Subspace_vectors_entry.stride
    // , nrows, 1 }; So, effectively the cols. are stored contiguously in
    // memory one after the other as Subspace_vectors_entry.stride = 1

    // to store vectors: u_i , i = 0, ..., subspace_dim -1 , we use matrix U
    const gko::batch_dense::BatchEntry<ValueType> U_entry{
        U, static_cast<size_type>(nrhs), nrows * static_cast<int>(subspace_dim),
        nrhs};
    // storage:row-major order , vectors corr. to each rhs
    // are stored in a single col. one after the other-(matrix U on paper).
    // And to get u_i : that is ith  vector for each rhs: u_i_entry{  &U[i*
    // U_entry.stride * nrows], U_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.

    // to store vectors: g_i , i = 0, ..., subspace_dim -1, we use matrix G
    const gko::batch_dense::BatchEntry<ValueType> G_entry{
        G, static_cast<size_type>(nrhs), nrows * static_cast<int>(subspace_dim),
        nrhs};
    // storage:row-major order , vectors corr. to each rhs
    // are stored in a single col. one after the other-(matrix G on paper).
    // And to get g_i : that is ith  vector for each rhs: g_i_entry{  &G[i*
    // G_entry.stride * nrows], G_entry.stride , nrows, nrhs}; So if nrhs=1,
    // effectively the cols. are stored contiguously in memory one after the
    // other.

    const gko::batch_dense::BatchEntry<ValueType> M_entry{
        M, subspace_dim * static_cast<size_type>(nrhs),
        static_cast<int>(subspace_dim), static_cast<int>(subspace_dim) * nrhs};
    // storage:row-major ,  entry (i,j) for different RHSs are placed one
    // after the other in a row - when drawn on paper, (and the same is true
    // for actual storage as the storage order is row-major) to get entry
    // (i,j) for rhs: rhs_k , scalar_M_i_j_for_rhs_k =  M[M_entry.stride*i +
    // j*nrhs  + rhs_k ]

    const gko::batch_dense::BatchEntry<ValueType> temp_for_single_rhs_entry{
        temp_for_single_rhs, static_cast<size_type>(1), nrows, 1};
    const gko::batch_dense::BatchEntry<ValueType> omega_entry{
        omega, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> temp1_entry{
        temp1, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<ValueType> temp2_entry{
        temp2, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> t_norms_entry{
        norms_t, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> r_norms_entry{
        norms_r, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
        norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
        norms_res, static_cast<size_type>(nrhs), 1, nrhs};
    const gko::batch_dense::BatchEntry<real_type> tmp_norms_entry{
        norms_tmp, static_cast<size_type>(subspace_dim), 1,
        static_cast<int>(subspace_dim)};

    // generate preconditioner
    prec.generate(ibatch, A_entry, prec_work);

    // initialization
    // compute b norms
    // r = b - A*x
    // compute residual norms
    // initialize G, U with zeroes
    // M = Identity
    // xs = x and rs = r if smoothing is enabled
    // initialize (either random numbers or deterministically) and
    // orthonormalize Subspace_vectors omega = 1
    initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry,
               G_entry, U_entry, M_entry, Subspace_vectors_entry, deterministic,
               xs_entry, rs_entry, smoothing, omega_entry, rhs_norms_entry,
               res_norms_entry, temp_for_single_rhs_entry, tmp_norms_entry);

    // stopping criterion object
    StopType stop(opts.residual_tol, rhs_norms_entry.values);

    int outer_iter = -1;

    // while (1) {
    //     outer_iter++;
    for (outer_iter = 0; outer_iter < opts.max_its; outer_iter++) {
        if (stop.check_converged(res_norms_entry.values)) {
            break;
        }

        // f = HermitianTranspose(P) * r
        update_f(gko::batch::to_const(Subspace_vectors_entry),
                 gko::batch::to_const(r_entry), subspace_dim, f_entry);

        for (size_type k = 0; k < subspace_dim; k++) {
            const gko::batch_dense::BatchEntry<ValueType> u_k_entry{
                &U_entry.values[k * nrows * U_entry.stride], U_entry.stride,
                nrows, nrhs};
            const gko::batch_dense::BatchEntry<ValueType> g_k_entry{
                &G_entry.values[k * nrows * G_entry.stride], G_entry.stride,
                nrows, nrhs};

            // solve c from Mc = f (Lower Triangular solve)
            update_c(gko::batch::to_const(M_entry),
                     gko::batch::to_const(f_entry), c_entry, temp1_entry);

            // v = r - ( c(k) * g_k  +  c(k+1) * g_(k+1)  + ...  +
            // c(subspace_dim - 1) * g_(subspace_dim - 1))
            update_v(gko::batch::to_const(G_entry),
                     gko::batch::to_const(c_entry),
                     gko::batch::to_const(r_entry), v_entry, k);

            // helper = v
            copy(gko::batch::to_const(v_entry), helper_entry);

            // v = precond * helper
            prec.apply(gko::batch::to_const(helper_entry), v_entry);

            // u_k = omega * v + (c(k) * u_k  +  c(k+1) * u_(k+1) + ...  +
            // c(subspace_dim - 1) * u_(subspace_dim - 1) )
            update_u_k(
                gko::batch::to_const(omega_entry),
                gko::batch::to_const(c_entry), gko::batch::to_const(v_entry),
                gko::batch::to_const(U_entry), k, helper_entry, u_k_entry);

            // g_k = A * u_k
            matvec_kernel(A_entry, gko::batch::to_const(u_k_entry), g_k_entry);

            // for i = 0 to k-1
            //     alpha = (p_i * g_k)/M(i,i)
            //     g_k = g_k - alpha * g_i
            //     u_k = u_k - alpha * u_i
            // end
            const gko::batch_dense::BatchEntry<ValueType>& alpha_entry =
                temp1_entry;
            update_g_k_and_u_k(k, gko::batch::to_const(G_entry),
                               gko::batch::to_const(U_entry),
                               gko::batch::to_const(Subspace_vectors_entry),
                               gko::batch::to_const(M_entry), alpha_entry,
                               g_k_entry, u_k_entry);

            // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
            update_M(gko::batch::to_const(g_k_entry),
                     gko::batch::to_const(Subspace_vectors_entry), M_entry, k);

            // beta = f(k)/M(k,k)
            const gko::batch_dense::BatchEntry<ValueType>& beta_entry =
                temp1_entry;
            beta_entry.values[0] = f_entry.values[k * f_entry.stride] /
                                   M_entry.values[k * M_entry.stride + k];

            // r = r - beta * g_k
            // x = x + beta * u_k
            update_r_and_x_inner_loop(gko::batch::to_const(g_k_entry),
                                      gko::batch::to_const(u_k_entry),
                                      gko::batch::to_const(beta_entry), r_entry,
                                      x_entry);

            if (smoothing == true) {
                const gko::batch_dense::BatchEntry<ValueType>& gamma_entry =
                    temp2_entry;
                smoothing_operation(gko::batch::to_const(x_entry),
                                    gko::batch::to_const(r_entry), gamma_entry,
                                    t_entry, xs_entry, rs_entry, t_norms_entry);
            }

            // if k + 1 <= subspace_dim - 1
            //     f(i) = 0 , where i = 0,...,k
            //     f(i) = f(i) - beta * M(i,k) ,where i = k + 1, ... ,
            //     subspace_dim -1
            // end if
            if (k + 1 <= subspace_dim - 1) {
                for (int row = 0; row <= k; row++) {
                    f_entry.values[row * f_entry.stride] = zero<ValueType>();
                }
                for (int row = k + 1; row < subspace_dim; row++) {
                    f_entry.values[row * f_entry.stride] -=
                        beta_entry.values[0] *
                        M_entry.values[row * M_entry.stride + k];
                }
            }
        }  // end of inner loop

        // v = precond * r
        prec.apply(gko::batch::to_const(r_entry), v_entry);

        // t = A *v
        matvec_kernel(A_entry, gko::batch::to_const(v_entry), t_entry);

        // omega = ( t * r )/ (t * t)
        // rho = (t * r ) /(||t|| * || r||)
        // if |rho| < kappa
        //      omega = omega * kappa / |rho|
        // end if
        const gko::batch_dense::BatchEntry<ValueType>& t_r_dot_entry =
            temp1_entry;
        const gko::batch_dense::BatchEntry<ValueType>& rho_entry = temp2_entry;
        compute_omega(gko::batch::to_const(t_entry),
                      gko::batch::to_const(r_entry), rho_entry, t_r_dot_entry,
                      t_norms_entry, r_norms_entry, omega_entry, kappa);

        // r = r - omega * t
        // x = x + omega * v
        update_r_and_x_outer_loop(
            gko::batch::to_const(t_entry), gko::batch::to_const(v_entry),
            gko::batch::to_const(omega_entry), r_entry, x_entry);

        if (smoothing == true) {
            const gko::batch_dense::BatchEntry<ValueType>& gamma_entry =
                temp2_entry;
            smoothing_operation(gko::batch::to_const(x_entry),
                                gko::batch::to_const(r_entry), gamma_entry,
                                t_entry, xs_entry, rs_entry, t_norms_entry);
            compute_norm2<ValueType>(gko::batch::to_const(rs_entry),
                                     res_norms_entry);
        } else {
            compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                     res_norms_entry);
        }
    }

    logger.log_iteration(ibatch, outer_iter, res_norms_entry.values[0]);

    if (smoothing == true) {
        copy(gko::batch::to_const(xs_entry), x_entry);
        copy(gko::batch::to_const(rs_entry), r_entry);
    }
}
