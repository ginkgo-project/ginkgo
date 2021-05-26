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

#include "core/solver/batch_gmres_kernels.hpp"


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
 * @brief The batch Gmres solver namespace.
 *
 * @ingroup batch_gmres
 */
namespace batch_gmres {

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

    // for(int vec_num  = 0; vec_num < restart + 1; vec_num++)
    // {
    //     for(int row_idx = 0; row_idx < num_rows ; row_idx++)
    //     {
    //         for(int rhs_idx = 0; rhs_idx < num_rhs ; rhs_idx++)
    //         {
    //             V_entry.values[vec_num * V_entry.stride*num_rows + row_idx *
    //             V_entry.stride + rhs_idx] = zero<ValueType>();
    //         }
    //     }
    // }

    // initialize H with zeroes
    for (int r = 0; r < H_entry.num_rows; r++) {
        for (int c = 0; c < H_entry.num_rhs; c++) {
            H_entry.values[r * H_entry.stride + c] = zero<ValueType>();
        }
    }

    // for(int row_idx = 0; row_idx < restart + 1; row_idx++)
    // {
    //     for(int col_idx = 0; col_idx < restart; col_idx++)
    //     {
    //         for(int rhs_idx = 0; rhs_idx < num_rhs ; rhs_idx++)
    //         {
    //             H_entry.values[row_idx * H_entry.stride + col_idx*num_rhs +
    //             rhs_idx] = zero<ValueType>();
    //         }
    //     }
    // }
}

template <typename ValueType>
inline void GeneratePlaneRotation(const ValueType *const x,
                                  const ValueType *const y, ValueType *const cs,
                                  ValueType *const sn, const int nrhs,
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
            ValueType temp = x[c] / y[c];
            sn[c] = one<ValueType>() / sqrt(one<ValueType>() + temp * temp);
            cs[c] = temp * sn[c];
        } else {
            ValueType temp = y[c] / x[c];
            cs[c] = one<ValueType>() / sqrt(one<ValueType>() + temp * temp);
            sn[c] = temp * cs[c];
        }
    }
}

template <typename ValueType>
inline void ApplyPlaneRotation(ValueType *const x, ValueType *const y,
                               const ValueType *const cs,
                               const ValueType *const sn, const int nrhs,
                               const uint32 &converged)
{
    for (int c = 0; c < nrhs; c++) {
        const uint32 conv = converged & (1 << c);

        if (conv) {
            continue;
        }

        ValueType temp = cs[c] * x[c] + sn[c] * y[c];
        y[c] = -one<ValueType>() * sn[c] * x[c] + cs[c] * y[c];
        x[c] = temp;
    }
}

template <typename ValueType>
inline void update_v_naught_and_s(
    const gko::batch_dense::BatchEntry<ValueType> &V_entry,
    const gko::batch_dense::BatchEntry<ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &z_entry,
    const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;

    const gko::batch_dense::BatchEntry<ValueType> v_naught_entry{
        V_entry.values, V_entry.stride, z_entry.num_rows, V_entry.num_rhs};


    real_type z_norms[batch_config<ValueType>::max_num_rhs];
    const gko::batch_dense::BatchEntry<real_type> z_norms_entry{
        z_norms, z_entry.stride, 1, z_entry.num_rhs};

    batch_dense::compute_norm2(z_entry, z_norms_entry);

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
    const PrecType &prec, const uint32 &converged)
{
    using real_type = typename gko::remove_complex<ValueType>;
    const int &i = inner_iter;
    const int num_rows = w_entry.num_rows;
    const int num_rhs = w_entry.num_rhs;

    const gko::batch_dense::BatchEntry<const ValueType> v_i_entry{
        &V_entry.values[i * V_entry.stride * num_rows], V_entry.stride,
        num_rows, V_entry.num_rhs};

    ValueType w_temp[batch_config<ValueType>::max_num_rows *
                     batch_config<ValueType>::max_num_rhs];

    const gko::batch_dense::BatchEntry<ValueType> w_temp_entry{
        w_temp, w_entry.stride, w_entry.num_rows, w_entry.num_rhs};

    spmv_kernel(A_entry, v_i_entry, w_temp_entry);

    prec.apply(gko::batch::to_const(w_temp_entry), w_entry);


    for (int k = 0; k <= i; k++) {
        const gko::batch_dense::BatchEntry<const ValueType> v_k_entry{
            &V_entry.values[k * V_entry.stride * num_rows], V_entry.stride,
            num_rows, V_entry.num_rhs};

        const gko::batch_dense::BatchEntry<ValueType> h_k_i_entry{
            &H_entry.values[k * H_entry.stride + i * num_rhs],
            static_cast<size_type>(w_entry.num_rhs), 1, w_entry.num_rhs};


        batch_dense::compute_dot_product(gko::batch::to_const(w_entry),
                                         v_k_entry, h_k_i_entry);


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


    real_type w_norms[batch_config<ValueType>::max_num_rhs];
    gko::batch_dense::BatchEntry<real_type> w_norms_entry{
        w_norms, w_entry.stride, 1, w_entry.num_rhs};

    batch_dense::compute_norm2(gko::batch::to_const(w_entry), w_norms_entry);

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
    const gko::batch_dense::BatchEntry<ValueType> &x_entry, const int m,
    const gko::batch_dense::BatchEntry<const ValueType> &H_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &s_entry,
    const gko::batch_dense::BatchEntry<const ValueType> &V_entry,
    const uint32 &converged)
{
    ValueType y[gko::kernels::batch_gmres::max_restart *
                batch_config<ValueType>::max_num_rhs];

    const gko::batch_dense::BatchEntry<ValueType> y_entry{
        y, static_cast<size_type>(x_entry.num_rhs), m + 1, x_entry.num_rhs};


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


}  // unnamed namespace


template <typename T>
using BatchGmresOptions = gko::kernels::batch_gmres::BatchGmresOptions<T>;

template <typename StopType, typename PrecType, typename LogType,
          typename BatchMatrixType, typename ValueType>
static void apply_impl(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchGmresOptions<remove_complex<ValueType>> &opts, LogType logger,
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
    const auto restart = opts.restart_num;

    const int local_size_bytes =
        gko::kernels::batch_gmres::local_memory_requirement<ValueType>(
            nrows, nrhs, restart) +
        PrecType::dynamic_work_size(nrows, a.num_nnz) * sizeof(ValueType);
    using byte = unsigned char;
    Array<byte> local_space(exec, local_size_bytes);

    for (size_type ibatch = 0; ibatch < nbatch; ibatch++) {
        byte *const shared_space = local_space.get_data();
        ValueType *const r = reinterpret_cast<ValueType *>(shared_space);
        ValueType *const z = r + nrows * nrhs;
        ValueType *const w = z + nrows * nrhs;
        ValueType *const cs = w + nrows * nrhs;
        ValueType *const sn = cs + restart * nrhs;
        ValueType *const s = sn + restart * nrhs;
        ValueType *const H = s + (restart + 1) * nrhs;  // Hessenberg matrix
        ValueType *const V = H + restart * (restart + 1) *
                                     nrhs;  // Krylov subspace basis vectors
        ValueType *const prec_work = V + nrows * (restart + 1) * nrhs;

        real_type *const norms_rhs = reinterpret_cast<real_type *>(
            prec_work + PrecType::dynamic_work_size(nrows, a.num_nnz));
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
            // storage:row-major , residual vector corresponding to each rhs is
            // stored as a col. of the matrix
            r, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> z_entry{
            // storage:row-major ,vector corresponding to each rhs is stored as
            // a col. of the matrix
            z, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> w_entry{
            // storage:row-major ,vector corresponding to each rhs is stored as
            // a col. of the matrix
            w, static_cast<size_type>(nrhs), nrows, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> cs_entry{
            // storage:row-major ,vector corresponding to each rhs is stored as
            // a col. of the matrix
            cs, static_cast<size_type>(nrhs), restart, nrhs};

        const gko::batch_dense::BatchEntry<ValueType> sn_entry{
            // storage:row-major ,vector corresponding to each rhs is stored as
            // a col. of the matrix
            sn, static_cast<size_type>(nrhs), restart, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> s_entry{
            // storage:row-major ,vector corresponding to each rhs is stored as
            // a col. of the matrix
            s, static_cast<size_type>(nrhs), restart + 1, nrhs};


        const gko::batch_dense::BatchEntry<ValueType> H_entry{
            H, static_cast<size_type>(nrhs * restart), restart + 1,
            restart * nrhs};
        // storage:row-major ,  entry (i,j) for different RHSs are placed after
        // the other in a row - when drawn on paper, (and the same is true for
        // actual storage as the storage order is row-major) to get entry (i,j)
        // for rhs: rhs_k , H_entry.stride*i + j*nrhs  + rhs_k

        const gko::batch_dense::BatchEntry<ValueType> V_entry{
            V, static_cast<size_type>(nrhs), nrows * (restart + 1), nrhs};
        // storage:row-major order , subspace basis vectors corr. to each rhs
        // are stored in a single col. one after the other-(on paper). And to
        // get vi : that is ith basis vector for each rhs: vi_entry{  &V[i*
        // V_entry.stride * nrows], V_entry.stride , nrows, nrhs}; So if nrhs=1,
        // effectively the cols. are stored contiguously in memory one after the
        // other.


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
        // z = precond*r
        // compute residual norms
        // initialize V,H,cs,sn with zeroes
        initialize(A_entry, b_entry, gko::batch::to_const(x_entry), r_entry,
                   prec, z_entry, cs_entry, sn_entry, V_entry, H_entry,
                   rhs_norms_entry, res_norms_entry);

        // stopping criterion object
        StopType stop(nrhs, opts.max_its, opts.abs_residual_tol,
                      opts.rel_residual_tol,
                      static_cast<stop::tolerance>(opts.tol_type), converged,
                      rhs_norms_entry.values);

        int outer_iter = -1;
        bool inner_loop_break_flag = false;

        while (1) {
            ++outer_iter;

            bool all_converged = stop.check_converged(
                outer_iter * (restart + 1), res_norms_entry.values,
                {NULL, 0, 0, 0}, converged);

            logger.log_iteration(ibatch, outer_iter * (restart + 1),
                                 res_norms_entry.values, converged);

            if (all_converged) {
                break;
            }


            // KrylovBasis_0 = z/norm(z)
            // s -> fill with zeroes
            // s(0) = norm(z)
            update_v_naught_and_s(V_entry, s_entry,
                                  gko::batch::to_const(z_entry), converged);

            for (int inner_iter = -1; inner_iter < restart - 1;) {
                inner_iter++;

                // w_temp = A * v_i
                // w = precond * w_temp
                // i = inner_iter
                // for k = 0 to inner_iter
                //     Hessenburg(k,i) =  w' * v_k
                //     w = w - Hessenburg(k,i) * v_k
                // end
                // Hessenburg(i+1, i) = norm(w)
                // KrylovBasis_i+1 = w / Hessenburg(i+1,i)
                arnoldi_method(A_entry, inner_iter, V_entry, H_entry, w_entry,
                               prec, converged);


                for (int k = 0; k < inner_iter; k++) {
                    // temp = cs(k) * Hessenberg( k, inner_iter )  +   sn(k) *
                    // Hessenberg(k + 1, inner_iter) Hessenberg(k + 1,
                    // inner_iter) = -1 * sn(k) * Hessenberg( k , inner_iter) +
                    // cs(k) * Hessenberg(k + 1 , inner_iter)
                    // Hessenberg(k,inner_iter) = temp
                    ApplyPlaneRotation(
                        &H_entry.values[k * H_entry.stride + inner_iter * nrhs],
                        &H_entry.values[(k + 1) * H_entry.stride +
                                        inner_iter * nrhs],
                        &cs_entry.values[k * cs_entry.stride],
                        &sn_entry.values[k * sn_entry.stride], nrhs, converged);
                }

                // compute sine and cos
                GeneratePlaneRotation(
                    &H_entry.values[inner_iter * H_entry.stride +
                                    inner_iter * nrhs],
                    &H_entry.values[(inner_iter + 1) * H_entry.stride +
                                    inner_iter * nrhs],
                    &cs_entry.values[inner_iter * cs_entry.stride],
                    &sn_entry.values[inner_iter * cs_entry.stride], nrhs,
                    converged);

                // temp = cs(inner_iter) * s(inner_iter)
                // s(inner_iter + 1) = -1 * sn(inner_iter) * s(inner_iter)
                // s(inner_iter) = temp
                // Hessenberg(inner_iter , inner_iter) = cs(inner_iter) *
                // Hessenberg(inner_iter , inner_iter) + sn(inner_iter) *
                // Hessenberg(inner_iter + 1, inner_iter) Hessenberg(inner_iter
                // + 1, inner_iter) = 0
                ApplyPlaneRotation(
                    &s_entry.values[inner_iter * s_entry.stride],
                    &s_entry.values[(inner_iter + 1) * s_entry.stride],
                    &cs_entry.values[inner_iter * cs_entry.stride],
                    &sn_entry.values[inner_iter * sn_entry.stride], nrhs,
                    converged);

                ApplyPlaneRotation(
                    &H_entry.values[inner_iter * H_entry.stride +
                                    inner_iter * nrhs],
                    &H_entry.values[(inner_iter + 1) * H_entry.stride +
                                    inner_iter * nrhs],
                    &cs_entry.values[inner_iter * cs_entry.stride],
                    &sn_entry.values[inner_iter * sn_entry.stride], nrhs,
                    converged);

                for (int c = 0; c < nrhs; c++) {
                    const uint32 conv = converged & (1 << c);

                    if (conv) {
                        continue;
                    }
                    H_entry.values[(inner_iter + 1) * H_entry.stride +
                                   inner_iter * nrhs + c] = zero<ValueType>();
                }


                // estimate of residual norms
                // residual = abs(s(inner_iter + 1))
                for (int c = 0; c < res_norms_temp_entry.num_rhs; c++) {
                    res_norms_temp_entry.values[c] = abs(
                        s_entry.values[(inner_iter + 1) * s_entry.stride + c]);
                }

                copy(gko::batch::to_const(res_norms_temp_entry),
                     res_norms_entry, converged);

                const uint32 converged_prev = converged;

                all_converged = stop.check_converged(
                    outer_iter * (restart + 1) + inner_iter + 1,
                    res_norms_entry.values, {NULL, 0, 0, 0}, converged);

                const uint32 converged_recent = converged_prev ^ converged;

                // y = Hessenburg(0 : inner_iter,0 : inner_iter) \ s(0 :
                // inner_iter) x = x + KrylovBasis(:, 0 : inner_iter ) * y
                update_x(x_entry, inner_iter, gko::batch::to_const(H_entry),
                         gko::batch::to_const(s_entry),
                         gko::batch::to_const(V_entry), ~converged_recent);

                logger.log_iteration(
                    ibatch, outer_iter * (restart + 1) + inner_iter + 1,
                    res_norms_entry.values, converged);

                if (all_converged) {
                    inner_loop_break_flag = true;
                    break;
                }
            }

            if (inner_loop_break_flag == true) {
                break;
            }

            // y = Hessenburg(0:restart - 1,0:restart - 1) \ s(0:restart-1)
            // x = x + KrylovBasis(:,0 : restart - 1) * y
            update_x(x_entry, restart - 1, gko::batch::to_const(H_entry),
                     gko::batch::to_const(s_entry),
                     gko::batch::to_const(V_entry), converged);


            // r = b
            copy(b_entry, r_entry, converged);

            // r = r - A*x
            advanced_spmv_kernel(static_cast<ValueType>(-1.0), A_entry,
                                 gko::batch::to_const(x_entry),
                                 static_cast<ValueType>(1.0), r_entry);
            batch_dense::compute_norm2<ValueType>(gko::batch::to_const(r_entry),
                                                  res_norms_temp_entry);

            copy(gko::batch::to_const(res_norms_temp_entry), res_norms_entry,
                 converged);

            // z = precond * r
            prec.apply(gko::batch::to_const(r_entry), z_entry);
        }


        if (left_entry.values) {
            batch_dense::batch_scale(right_entry, x_entry);
        }
    }
}


template <typename BatchType, typename LoggerType, typename ValueType>
void apply_select_prec(
    std::shared_ptr<const ReferenceExecutor> exec,
    const BatchGmresOptions<remove_complex<ValueType>> &opts,
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
           const BatchGmresOptions<remove_complex<ValueType>> &opts,
           const BatchLinOp *const a,
           const matrix::BatchDense<ValueType> *const left_scale,
           const matrix::BatchDense<ValueType> *const right_scale,
           const matrix::BatchDense<ValueType> *const b,
           matrix::BatchDense<ValueType> *const x,
           gko::log::BatchLogData<ValueType> &logdata)
{
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

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_GMRES_APPLY_KERNEL);


}  // namespace batch_gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
