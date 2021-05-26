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

#include "core/solver/batch_idr_kernels.hpp"

#include <ctime>
#include <random>


#include "reference/base/config.hpp"
// include device kernels for every matrix and preconditioner type
#include "reference/log/batch_logger.hpp"
#include "reference/matrix/batch_csr_kernels.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_identity.hpp"
#include "reference/preconditioner/batch_jacobi.hpp"
#include "reference/stop/batch_criteria.hpp"


namespace gko {
namespace kernels {
namespace reference {


/**
 * @brief The batch Idr solver namespace.
 *
 * @ingroup batch_idr
 */
namespace batch_idr {

namespace {

template <typename ValueType>
inline void copy(
    const gko::batch_dense::BatchEntry<const ValueType> &source_entry,
    const gko::batch_dense::BatchEntry<ValueType> &destination_entry,
    const uint32 &converged)
{
    for (int r = 0; r < source_entry.num_rows; r++) {
        for (int c = 0; c < source_entry.num_rhs; c++) {
            const uint32 conv = converged & (1 << c);

            if (conv) {
                continue;
            }

            destination_entry.values[r * destination_entry.stride + c] =
                source_entry.values[r * source_entry.stride + c];
        }
    }
}


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
    const int num_rows, const int subspace_dim)
{
    using real_type = typename gko::remove_complex<ValueType>;

    real_type norms[gko::kernels::batch_idr::max_subspace_dim];

    const gko::batch_dense::BatchEntry<real_type> norms_entry{
        norms, static_cast<size_type>(subspace_dim), 1, subspace_dim};

    for (int i = 0; i < subspace_dim; i++) {
        const gko::batch_dense::BatchEntry<ValueType> p_i_entry{
            &Subspace_vectors_entry
                 .values[i * num_rows * Subspace_vectors_entry.stride],
            Subspace_vectors_entry.stride, num_rows, 1};

        ValueType w_i[batch_config<ValueType>::max_num_rows];

        const gko::batch_dense::BatchEntry<ValueType> w_i_entry{w_i, 1,
                                                                num_rows, 1};

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
                norms_entry.values[j] * norms_entry.values[j]);

            mul_entry.values[0] *= -one<ValueType>();
            batch_dense::add_scaled(gko::batch::to_const(mul_entry),
                                    gko::batch::to_const(w_j_entry), w_i_entry);
        }

        // p_i = w_i
        batch_dense::copy(gko::batch::to_const(w_i_entry), p_i_entry);

        batch_dense::compute_norm2(gko::batch::to_const(w_i_entry),
                                   gko::batch_dense::BatchEntry<real_type>{
                                       &norms_entry.values[i], 1, 1, 1});
    }

    // e_k = w_k / || w_k ||
    for (int k = 0; k < subspace_dim; k++) {
        const gko::batch_dense::BatchEntry<ValueType> w_k_entry{
            &Subspace_vectors_entry
                 .values[k * num_rows * Subspace_vectors_entry.stride],
            Subspace_vectors_entry.stride, num_rows, 1};

        ValueType scale_factor =
            one<ValueType>() / static_cast<ValueType>(norms_entry.values[k]);

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
        &res_norms_entry)
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
            for (int rhs_index = 0; rhs_index < num_rows; rhs_index++) {
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
                                    subspace_dim);
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
    const uint32 &converged)
{
    const auto subspace_dim = M_entry.num_rows;
    const auto nrhs = f_entry.num_rhs;
    // upper triangular solve
    // solve top to bottom
    for (int row_index = 0; row_index < subspace_dim; row_index++) {
        ValueType temp_sum[batch_config<ValueType>::max_num_rhs] = {
            zero<ValueType>()};

        for (int col_index = 0; col_index < row_index; col_index++) {
            for (int rhs_index = 0; rhs_index < nrhs; rhs_index++) {
                const uint32 conv = converged & (1 << rhs_index);

                if (conv) {
                    continue;
                }
                temp_sum[rhs_index] +=
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
                 temp_sum[rhs_index]) /
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
    copy(gko::batch::to_const(r_entry), v_entry, converged);

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
    const size_type k,
    const gko::batch_dense::BatchEntry<ValueType> &helper_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_entry,
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

    copy(gko::batch::to_const(helper_entry), u_k_entry, converged);
}


template <typename ValueType>
inline void update_g_k_and_u_k(
    const size_type k,
    const gko::batch_dense::BatchEntry<ValueType> &alpha_entry,
    const gko::batch_dense::BatchEntry<ValueType> &g_k_entry,
    const gko::batch_dense::BatchEntry<ValueType> &u_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &G_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &U_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &Subspace_vectors_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &M_entry,
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

                g_k_entry.values[row * g_k_entry.stride + rhs] -=
                    alpha_entry.values[rhs] *
                    G_entry.values[i * nrows * G_entry.stride +
                                   row * G_entry.stride + rhs];

                u_k_entry.values[row * u_k_entry.stride + rhs] -=
                    alpha_entry.values[rhs] *
                    U_entry.values[i * nrows * U_entry.stride +
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
inline void update_r_inner_loop(
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &g_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_entry,
    const uint32 &converged)
{
    for (int row = 0; row < g_k_entry.num_rows; row++) {
        for (int rhs = 0; rhs < g_k_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            r_entry.values[row * r_entry.stride + rhs] -=
                beta_entry.values[rhs] *
                g_k_entry.values[row * g_k_entry.stride + rhs];
        }
    }
}


template <typename ValueType>
inline void update_x_inner_loop(
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &u_k_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &beta_entry,
    const uint32 &converged)
{
    for (int row = 0; row < u_k_entry.num_rows; row++) {
        for (int rhs = 0; rhs < u_k_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            x_entry.values[row * x_entry.stride + rhs] +=
                beta_entry.values[rhs] *
                u_k_entry.values[row * u_k_entry.stride + rhs];
        }
    }
}


template <typename ValueType>
inline void compute_omega(
    const gko::batch_dense::BatchEntry<ValueType> &omega_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_entry,
    const gko::batch_dense::BatchEntry<ValueType> &t_r_dot_entry,
    const gko::batch_dense::BatchEntry<gko::remove_complex<ValueType>>
        &norms_t_entry,
    const gko::remove_complex<ValueType> kappa, const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;

    batch_dense::compute_dot_product(gko::batch::to_const(t_entry),
                                     gko::batch::to_const(r_entry),
                                     t_r_dot_entry);

    batch_dense::compute_norm2(gko::batch::to_const(t_entry), norms_t_entry);

    real_type norms_r[batch_config<ValueType>::max_num_rhs];
    const auto nrhs = omega_entry.num_rhs;
    const gko::batch_dense::BatchEntry<real_type> norms_r_entry{
        norms_r, nrhs, static_cast<int>(1), static_cast<int>(nrhs)};
    batch_dense::compute_norm2(gko::batch::to_const(r_entry), norms_r_entry);

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
inline void update_r_outer_loop(
    const gko::batch_dense::BatchEntry<ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const uint32 &converged)
{
    for (int row = 0; row < t_entry.num_rows; row++) {
        for (int rhs = 0; rhs < t_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            r_entry.values[row * r_entry.stride + rhs] -=
                omega_entry.values[rhs] *
                t_entry.values[row * t_entry.stride + rhs];
        }
    }
}


template <typename ValueType>
inline void update_x_outer_loop(
    const gko::batch_dense::BatchEntry<ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &v_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &omega_entry,
    const uint32 &converged)
{
    for (int row = 0; row < v_entry.num_rows; row++) {
        for (int rhs = 0; rhs < v_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            x_entry.values[row * x_entry.stride + rhs] +=
                omega_entry.values[rhs] *
                v_entry.values[row * v_entry.stride + rhs];
        }
    }
}

template <typename ValueType>
inline void smoothing_operation(
    const gko::batch_dense::BatchEntry<ValueType> &t_entry,
    const gko::batch_dense::BatchEntry<ValueType> &xs_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rs_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &x_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &r_entry,
    const gko::batch_dense::BatchEntry<ValueType> &rho_entry,
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


    // rho = (t * rs)/(t * t)
    batch_dense::compute_dot_product(gko::batch::to_const(t_entry),
                                     gko::batch::to_const(rs_entry), rho_entry);
    batch_dense::compute_norm2(gko::batch::to_const(t_entry), norms_t_entry);

    for (int rhs = 0; rhs < t_entry.num_rhs; rhs++) {
        rho_entry.values[rhs] /= static_cast<ValueType>(
            norms_t_entry.values[rhs] * norms_t_entry.values[rhs]);
    }


    // rs = rs - rho*(rs - r)
    // xs = xs - rho*(xs - x)
    for (int row = 0; row < rs_entry.num_rows; row++) {
        for (int rhs = 0; rhs < rs_entry.num_rhs; rhs++) {
            const uint32 conv = converged & (1 << rhs);

            if (conv) {
                continue;
            }

            rs_entry.values[row * rs_entry.stride + rhs] =
                (one<ValueType>() - rho_entry.values[rhs]) *
                    rs_entry.values[row * rs_entry.stride + rhs] +
                rho_entry.values[rhs] *
                    r_entry.values[row * r_entry.stride + rhs];

            xs_entry.values[row * xs_entry.stride + rhs] =
                (one<ValueType>() - rho_entry.values[rhs]) *
                    xs_entry.values[row * xs_entry.stride + rhs] +
                rho_entry.values[rhs] *
                    x_entry.values[row * x_entry.stride + rhs];
        }
    }
}

}  // unnamed namespace

template <typename T>
using BatchIdrOptions = gko::kernels::batch_idr::BatchIdrOptions<T>;


template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchIdrOptions<remove_complex<ValueType>> &opts, LogType logger,
    PrecType prec, const BatchMatrixType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const size_type nbatch = a.num_batch;
    const auto nrows = a.num_rows;
    const auto nrhs = b.num_rhs;
    const auto subspace_dim = opts.subspace_dim_val;
    const auto kappa = opts.kappa_val;
    const auto smoothing = opts.to_use_smoothing;
    const auto deterministic = opts.deterministic_gen;

    const int local_size_bytes =
        gko::kernels::batch_idr::local_memory_requirement<ValueType>(
            nrows, nrhs, subspace_dim) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    using byte = unsigned char;
    Array<byte> local_space(exec, local_size_bytes);


    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        byte *const shared_space = local_space.get_data();
        ValueType *const r = reinterpret_cast<ValueType *>(shared_space);
        ValueType *const t = r + nrows * nrhs;
        ValueType *const v = t + nrows * nrhs;
        ValueType *const xs = v + nrows * nrhs;
        ValueType *const rs = xs + nrows * nrhs;
        ValueType *const helper = rs + nrows * nrhs;
        ValueType *const f = helper + nrows * nrhs;
        ValueType *const c = f + subspace_dim * nrhs;
        ValueType *const Subspace_vectors = c + subspace_dim * nrhs;
        ValueType *const G = Subspace_vectors + nrows * subspace_dim;
        ValueType *const U = G + nrows * subspace_dim * nrhs;
        ValueType *const M = U + nrows * subspace_dim * nrhs;
        ValueType *const prec_work = M + subspace_dim * subspace_dim * nrhs;
        ;
        ValueType *const omega =
            prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz);
        ValueType *const alpha = omega + nrhs;
        ValueType *const beta = alpha + nrhs;
        ValueType *const rho = beta + nrhs;
        ValueType *const t_r_dot = rho + nrhs;
        real_type *const norms_t =
            reinterpret_cast<real_type *>(t_r_dot + nrhs);
        real_type *const norms_rhs = norms_t + nrhs;
        real_type *const norms_res = norms_rhs + nrhs;
        real_type *const norms_res_temp = norms_res + nrhs;


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
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> t_entry{
            t, static_cast<size_type>(nrhs), nrows, nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> v_entry{
            v, static_cast<size_type>(nrhs), nrows, nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> helper_entry{
            helper, static_cast<size_type>(nrhs), nrows, nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> xs_entry{
            xs, static_cast<size_type>(nrhs), nrows, nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> rs_entry{
            rs, static_cast<size_type>(nrhs), nrows, nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> f_entry{
            f, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim),
            nrhs};
        // storage:row-major , residual vector corresponding to each rhs is
        // stored as a col. of the matrix

        const gko::batch_dense::BatchEntry<ValueType> c_entry{
            c, static_cast<size_type>(nrhs), static_cast<int>(subspace_dim),
            nrhs};
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
            U, static_cast<size_type>(nrhs),
            nrows * static_cast<int>(subspace_dim), nrhs};
        // storage:row-major order , vectors corr. to each rhs
        // are stored in a single col. one after the other-(matrix U on paper).
        // And to get u_i : that is ith  vector for each rhs: u_i_entry{  &U[i*
        // U_entry.stride * nrows], U_entry.stride , nrows, nrhs}; So if nrhs=1,
        // effectively the cols. are stored contiguously in memory one after the
        // other.

        // to store vectors: g_i , i = 0, ..., subspace_dim -1, we use matrix G
        const gko::batch_dense::BatchEntry<ValueType> G_entry{
            G, static_cast<size_type>(nrhs),
            nrows * static_cast<int>(subspace_dim), nrhs};
        // storage:row-major order , vectors corr. to each rhs
        // are stored in a single col. one after the other-(matrix G on paper).
        // And to get g_i : that is ith  vector for each rhs: g_i_entry{  &G[i*
        // G_entry.stride * nrows], G_entry.stride , nrows, nrhs}; So if nrhs=1,
        // effectively the cols. are stored contiguously in memory one after the
        // other.


        const gko::batch_dense::BatchEntry<ValueType> M_entry{
            M, subspace_dim * static_cast<size_type>(nrhs),
            static_cast<int>(subspace_dim),
            static_cast<int>(subspace_dim) * nrhs};
        // storage:row-major ,  entry (i,j) for different RHSs are placed one
        // after the other in a row - when drawn on paper, (and the same is true
        // for actual storage as the storage order is row-major) to get entry
        // (i,j) for rhs: rhs_k , scalar_M_i_j_for_rhs_k =  M[M_entry.stride*i +
        // j*nrhs  + rhs_k ]


        const gko::batch_dense::BatchEntry<ValueType> omega_entry{
            omega, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> alpha_entry{
            alpha, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> beta_entry{
            beta, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> rho_entry{
            rho, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> t_r_dot_entry{
            t_r_dot, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> t_norms_entry{
            norms_t, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> rhs_norms_entry{
            norms_rhs, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> res_norms_entry{
            norms_res, static_cast<size_type>(nrhs), 1, nrhs};


        const gko::batch_dense::BatchEntry<real_type> res_norms_temp_entry{
            norms_res_temp, static_cast<size_type>(nrhs), 1, nrhs};

        // generate preconditioner
        prec.generate(A_entry, prec_work);

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
                   G_entry, U_entry, M_entry, Subspace_vectors_entry,
                   deterministic, xs_entry, rs_entry, smoothing, omega_entry,
                   rhs_norms_entry, res_norms_entry);


        // stopping criterion object
        StopType stop(nrhs, opts.max_its, opts.abs_residual_tol,
                      opts.rel_residual_tol,
                      static_cast<stop::tolerance>(opts.tol_type), converged,
                      rhs_norms_entry.values);

        int outer_iter = -1;

        while (1) {
            outer_iter++;


            bool all_converged = stop.check_converged(
                outer_iter, res_norms_entry.values, {NULL, 0, 0, 0}, converged);

            logger.log_iteration(ibatch, outer_iter, res_norms_entry.values,
                                 converged);

            if (all_converged) {
                break;
            }

            // f = HermitianTranspose(P) * r
            update_f(gko::batch::to_const(Subspace_vectors_entry),
                     gko::batch::to_const(r_entry), subspace_dim, f_entry,
                     converged);


            for (size_type k = 0; k < subspace_dim; k++) {
                const gko::batch_dense::BatchEntry<ValueType> u_k_entry{
                    &U_entry.values[k * nrows * U_entry.stride], U_entry.stride,
                    nrows, nrhs};

                const gko::batch_dense::BatchEntry<ValueType> g_k_entry{
                    &G_entry.values[k * nrows * G_entry.stride], G_entry.stride,
                    nrows, nrhs};


                // solve c from Mc = f (Lower Triangular solve)
                update_c(gko::batch::to_const(M_entry),
                         gko::batch::to_const(f_entry), c_entry, converged);

                // v = r - ( c(k) * g_k  +  c(k+1) * g_(k+1)  + ...  +
                // c(subspace_dim - 1) * g_(subspace_dim - 1))
                update_v(gko::batch::to_const(G_entry),
                         gko::batch::to_const(c_entry),
                         gko::batch::to_const(r_entry), v_entry, k, converged);


                // helper = v
                copy(gko::batch::to_const(v_entry), helper_entry, converged);

                // v = precond * helper
                prec.apply(gko::batch::to_const(helper_entry), v_entry);


                // u_k = omega * v + (c(k) * u_k  +  c(k+1) * u_(k+1) + ...  +
                // c(subspace_dim - 1) * u_(subspace_dim - 1) )
                update_u_k(gko::batch::to_const(omega_entry),
                           gko::batch::to_const(c_entry),
                           gko::batch::to_const(v_entry), k, helper_entry,
                           gko::batch::to_const(U_entry), u_k_entry, converged);


                // g_k = A * u_k
                spmv_kernel(A_entry, gko::batch::to_const(u_k_entry),
                            g_k_entry);


                // for i = 0 to k-1
                //     alpha = (p_i * g_k)/M(i,i)
                //     g_k = g_k - alpha * g_i
                //     u_k = u_k - alpha * u_i
                // end
                update_g_k_and_u_k(k, alpha_entry, g_k_entry, u_k_entry,
                                   gko::batch::to_const(G_entry),
                                   gko::batch::to_const(U_entry),
                                   gko::batch::to_const(Subspace_vectors_entry),
                                   gko::batch::to_const(M_entry), converged);


                // M(i,k) = p_i * g_k where i = k , k + 1, ... , subspace_dim -1
                update_M(gko::batch::to_const(g_k_entry),
                         gko::batch::to_const(Subspace_vectors_entry), M_entry,
                         k, converged);


                // beta = f(k)/M(k,k)
                for (int rhs = 0; rhs < nrhs; rhs++) {
                    const uint32 conv = converged & (1 << rhs);

                    if (conv) {
                        continue;
                    }

                    beta_entry.values[rhs] =
                        f_entry.values[k * f_entry.stride + rhs] /
                        M_entry.values[k * M_entry.stride + k * nrhs + rhs];
                }


                // r = r - beta * g_k
                update_r_inner_loop(r_entry, gko::batch::to_const(g_k_entry),
                                    gko::batch::to_const(beta_entry),
                                    converged);


                // x = x + beta * u_k
                update_x_inner_loop(x_entry, gko::batch::to_const(u_k_entry),
                                    gko::batch::to_const(beta_entry),
                                    converged);


                if (smoothing == true) {
                    smoothing_operation(t_entry, xs_entry, rs_entry,
                                        gko::batch::to_const(x_entry),
                                        gko::batch::to_const(r_entry),
                                        rho_entry, t_norms_entry, converged);
                }


                // if k + 1 <= subspace_dim - 1
                //     f(i) = 0 , where i = 0,...,k
                //     f(i) = f(i) - beta * M(i,k) ,where i = k + 1, ... ,
                //     subspace_dim -1
                // end if
                if (k + 1 <= subspace_dim - 1) {
                    for (int row = 0; row <= k; row++) {
                        for (int rhs = 0; rhs < nrhs; rhs++) {
                            f_entry.values[row * f_entry.stride + rhs] =
                                zero<ValueType>();
                        }
                    }

                    for (int row = k + 1; row < subspace_dim; row++) {
                        for (int rhs = 0; rhs < nrhs; rhs++) {
                            f_entry.values[row * f_entry.stride + rhs] -=
                                beta_entry.values[rhs] *
                                M_entry.values[row * M_entry.stride + k * nrhs +
                                               rhs];
                        }
                    }
                }


            }  // end of inner loop


            // v = precond * r
            prec.apply(gko::batch::to_const(r_entry), v_entry);

            // t = A *v
            spmv_kernel(A_entry, gko::batch::to_const(v_entry), t_entry);


            // omega = ( t * r )/ (t * t)
            // rho = (t * r ) /(||t|| * || r||)
            // if |rho| < kappa
            //      omega = omega * kappa / |rho|
            // end if
            compute_omega(omega_entry, gko::batch::to_const(t_entry),
                          gko::batch::to_const(r_entry), rho_entry,
                          t_r_dot_entry, t_norms_entry, kappa, converged);


            // r = r - omega * t
            update_r_outer_loop(r_entry, gko::batch::to_const(t_entry),
                                gko::batch::to_const(omega_entry), converged);

            // x = x + omega * v
            update_x_outer_loop(x_entry, gko::batch::to_const(v_entry),
                                gko::batch::to_const(omega_entry), converged);


            if (smoothing == true) {
                smoothing_operation(t_entry, xs_entry, rs_entry,
                                    gko::batch::to_const(x_entry),
                                    gko::batch::to_const(r_entry), rho_entry,
                                    t_norms_entry, converged);

                batch_dense::compute_norm2<ValueType>(
                    gko::batch::to_const(rs_entry),
                    res_norms_temp_entry);  // store residual norms in temp
                                            // entry

                copy(gko::batch::to_const(res_norms_temp_entry),
                     res_norms_entry,
                     converged);  // copy into res_norms entry only for those
                                  // RHSs which have not yet converged.
            } else {
                batch_dense::compute_norm2<ValueType>(
                    gko::batch::to_const(r_entry),
                    res_norms_temp_entry);  // store residual norms in temp
                                            // entry

                copy(gko::batch::to_const(res_norms_temp_entry),
                     res_norms_entry,
                     converged);  // copy into res_norms entry only for those
                                  // RHSs which have not yet converged.
            }
        }

        if (smoothing == true) {
            copy(gko::batch::to_const(xs_entry), x_entry, 0x00000000);
            copy(gko::batch::to_const(rs_entry), r_entry, 0x00000000);
        }

        if (left_entry.values) {
            batch_dense::batch_scale(right_entry, x_entry);
        }
    }
}

template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchIdrOptions<remove_complex<ValueType>> &opts,
    const LoggerType logger, const BatchType &a,
    const gko::batch_dense::UniformBatch<const ValueType> &left,
    const gko::batch_dense::UniformBatch<const ValueType> &right,
    const gko::batch_dense::UniformBatch<ValueType> &b,
    const gko::batch_dense::UniformBatch<ValueType> &x)
{
    if (opts.preconditioner == gko::preconditioner::batch::type::none) {
        BatchIdentity<ValueType> prec;

        apply_impl<stop::AbsAndRelResidualMaxIter<ValueType>>(
            exec, opts, logger, prec, a, left, right, b, x);

    } else if (opts.preconditioner ==
               gko::preconditioner::batch::type::jacobi) {
        BatchJacobi<ValueType> prec;
        apply_impl<stop::AbsAndRelResidualMaxIter<ValueType>>(
            exec, opts, logger, prec, a, left, right, b, x);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const BatchIdrOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
    if (opts.is_complex_subspace == true &&
        !is_complex<ValueType>())  // Currently, the option of having complex
                                   // subspace for real matrices is not
                                   // supported.
    {
        GKO_NOT_IMPLEMENTED;
    }

    batch_log::FinalLogger<remove_complex<ValueType>> logger(
        b->get_size().at(0)[1], opts.max_its, logdata.res_norms->get_values(),
        logdata.iter_counts.get_data());

    const gko::batch_dense::UniformBatch<const ValueType> b_b =
        get_batch_struct(b);

    const gko::batch_dense::UniformBatch<const ValueType> left_sb =
        maybe_null_batch_struct(left_scale);
    const gko::batch_dense::UniformBatch<const ValueType> right_sb =
        maybe_null_batch_struct(right_scale);
    const auto to_scale = left_sb.values || right_sb.values;
    if (to_scale) {
        if (!left_sb.values || !right_sb.values) {
            // one-sided scaling not implemented
            GKO_NOT_IMPLEMENTED;
        }
    }

    const gko::batch_dense::UniformBatch<ValueType> x_b = get_batch_struct(x);
    if (auto a_mat = dynamic_cast<const matrix::BatchCsr<ValueType> *>(a)) {
        // if(to_scale) {
        // We pinky-promise not to change the matrix and RHS if no scaling was
        // requested
        const gko::batch_csr::UniformBatch<ValueType> a_b =
            get_batch_struct(const_cast<matrix::BatchCsr<ValueType> *>(a_mat));
        const gko::batch_dense::UniformBatch<ValueType> b_b =
            get_batch_struct(const_cast<matrix::BatchDense<ValueType> *>(b));
        apply_select_prec(exec, opts, logger, a_b, left_sb, right_sb, b_b, x_b);
        // } else {
        // 	const gko::batch_csr::UniformBatch<const ValueType> a_b =
        // get_batch_struct(a_mat); 	apply_select_prec(exec, opts, logger,
        // a_b, left_sb, right_sb, &b_b, b_b, x_b);
        // }

    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDR_APPLY_KERNEL);


}  // namespace batch_idr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
