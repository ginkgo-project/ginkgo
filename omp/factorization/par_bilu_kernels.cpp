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

#include "core/factorization/par_bilu_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>


#include "reference/components/fixed_block_operations.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The parallel iterative block ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace par_bilu_factorization {


template <int bs, bool jacobi, typename ValueType, typename IndexType>
static inline void bilu_kernel(
    const IndexType nbrows, const IndexType *const row_ptrs,
    const IndexType *const col_idxs, const ValueType *const values,
    const IndexType *const row_ptrs_l, const IndexType *const col_idxs_l,
    ValueType *const l_values, const IndexType *const col_ptrs_u,
    const IndexType *const row_idxs_u, ValueType *const u_values,
    ValueType *const n_l_values = nullptr,
    ValueType *const n_u_values = nullptr)
{
    using Blk_t = blockutils::FixedBlock<ValueType, bs, bs>;
    const auto vals = reinterpret_cast<const Blk_t *>(values);
    const auto vals_l = reinterpret_cast<Blk_t *>(l_values);
    const auto vals_ut = reinterpret_cast<Blk_t *>(u_values);
    const auto n_vals_l = reinterpret_cast<Blk_t *>(n_l_values);
    const auto n_vals_ut = reinterpret_cast<Blk_t *>(n_u_values);
#pragma omp parallel for default(shared)
    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
             ibz++) {
            const auto jbcol = col_idxs[ibz];
            auto lbz = row_ptrs_l[ibrow];
            auto ubz = col_ptrs_u[jbcol];

            Blk_t sum;
            Blk_t last_op;
            ValueType *const sumarr = &sum(0, 0);
            ValueType *const lastarr = &last_op(0, 0);
            const ValueType *const valarr = &vals[ibz](0, 0);
#pragma omp simd
            for (int i = 0; i < bs * bs; i++) {
                sumarr[i] = valarr[i];
                lastarr[i] = zero<ValueType>();
            }

            // Calculate: sum = system_matrix(row, col) -
            //   dot(l_factor(row, :), u_factor(:, col))
            while (lbz < row_ptrs_l[ibrow + 1] && ubz < col_ptrs_u[jbcol + 1]) {
                const auto bcol_l = col_idxs_l[lbz];
                const auto brow_u = row_idxs_u[ubz];
                if (bcol_l == brow_u) {
                    // last_op = vals_l[row_l] * vals_u[row_u];
                    // sum -= last_op;
#pragma omp simd collapse(2)
                    for (int j = 0; j < bs; j++) {
                        for (int i = 0; i < bs; i++) {
                            last_op(i, j) = zero<ValueType>();
                            for (int k = 0; k < bs; k++) {
                                last_op(i, j) +=
                                    vals_l[lbz](i, k) * vals_ut[ubz](j, k);
                            }
                            sum(i, j) -= last_op(i, j);
                        }
                    }
                    lbz++;
                    ubz++;
                } else if (bcol_l < brow_u) {
                    ++lbz;
                } else {
                    ++ubz;
                }
            }

            // undo the last operation
#pragma omp simd collapse(2)
            for (int j = 0; j < bs; j++) {
                for (int i = 0; i < bs; i++) {
                    sum(i, j) += last_op(i, j);
                }
            }

            if (ibrow > jbcol) {
                // modify entry in L
                Blk_t invU;
#pragma omp simd collapse(2)
                for (int j = 0; j < bs; j++) {
                    for (int i = 0; i < bs; i++) {
                        invU(i, j) = vals_ut[col_ptrs_u[jbcol + 1] - 1](i, j);
                    }
                }

                const bool invflag = invert_block_complete(invU);
                if (!invflag) {
                    printf(" Could not invert diag block at blk row %ld!",
                           static_cast<long int>(ibrow));
                }

#pragma omp simd collapse(2)
                for (int j = 0; j < bs; j++) {
                    for (int i = 0; i < bs; i++) {
                        if (jacobi) {
                            n_vals_l[lbz - 1](i, j) = 0;
                        } else {
                            vals_l[lbz - 1](i, j) = 0;
                        }
                        for (int k = 0; k < bs; k++) {
                            if (jacobi) {
                                n_vals_l[lbz - 1](i, j) +=
                                    sum(i, k) * invU(j, k);
                            } else {
                                vals_l[lbz - 1](i, j) += sum(i, k) * invU(j, k);
                            }
                        }
                    }
                }
            } else {
                // modify entry in U
#pragma omp simd collapse(2)
                for (int j = 0; j < bs; j++) {
                    for (int i = 0; i < bs; i++) {
                        if (jacobi) {
                            n_vals_ut[ubz - 1](i, j) = sum(j, i);
                        } else {
                            vals_ut[ubz - 1](i, j) = sum(j, i);
                        }
                    }
                }
            }
        }
    }
}


template <int bs, typename ValueType, typename IndexType>
static void compute_bilu_impl(
    const std::shared_ptr<const OmpExecutor> exec, const int iters,
    const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor_t)
{
    const IndexType nbrows = sysmat->get_num_block_rows();
    const IndexType *const col_idxs = sysmat->get_const_col_idxs();
    const IndexType *const row_ptrs = sysmat->get_const_row_ptrs();
    const IndexType *const row_ptrs_l = l_factor->get_const_row_ptrs();
    const IndexType *const col_ptrs_u = u_factor_t->get_const_row_ptrs();
    const IndexType *const col_idxs_l = l_factor->get_const_col_idxs();
    const IndexType *const row_idxs_u = u_factor_t->get_const_col_idxs();
    const ValueType *const values = sysmat->get_const_values();
    ValueType *const l_values = l_factor->get_values();
    ValueType *const u_values = u_factor_t->get_values();

    for (int sweep = 0; sweep < iters; ++sweep) {
        bilu_kernel<bs, false>(nbrows, row_ptrs, col_idxs, values, row_ptrs_l,
                               col_idxs_l, l_values, col_ptrs_u, row_idxs_u,
                               u_values);
    }
}


template <typename ValueType, typename IndexType>
void compute_bilu_factors(
    const std::shared_ptr<const OmpExecutor> exec, const int iters,
    const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
    matrix::Fbcsr<ValueType, IndexType> *const lfactor,
    matrix::Fbcsr<ValueType, IndexType> *const ufactor)
{
    const int bs = sysmat->get_block_size();
    GKO_ASSERT(bs == lfactor->get_block_size());
    GKO_ASSERT(bs == ufactor->get_block_size());

    if (bs == 2) {
        compute_bilu_impl<2, ValueType, IndexType>(exec, iters, sysmat, lfactor,
                                                   ufactor);
    } else if (bs == 3) {
        compute_bilu_impl<3, ValueType, IndexType>(exec, iters, sysmat, lfactor,
                                                   ufactor);
    } else if (bs == 4) {
        compute_bilu_impl<4, ValueType, IndexType>(exec, iters, sysmat, lfactor,
                                                   ufactor);
    } else if (bs == 7) {
        compute_bilu_impl<7, ValueType, IndexType>(exec, iters, sysmat, lfactor,
                                                   ufactor);
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           " block size = " + std::to_string(bs));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_BILU_FACTORS_FBCSR_KERNEL);


template <int bs, typename ValueType, typename IndexType>
static void compute_bilu_jac_impl(
    const std::shared_ptr<const OmpExecutor> exec, const int iters,
    const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor_t)
{
    const IndexType nbrows = sysmat->get_num_block_rows();
    const IndexType *const col_idxs = sysmat->get_const_col_idxs();
    const IndexType *const row_ptrs = sysmat->get_const_row_ptrs();
    const IndexType *const row_ptrs_l = l_factor->get_const_row_ptrs();
    const IndexType *const col_ptrs_u = u_factor_t->get_const_row_ptrs();
    const IndexType *const col_idxs_l = l_factor->get_const_col_idxs();
    const IndexType *const row_idxs_u = u_factor_t->get_const_col_idxs();

    using Blk_t = blockutils::FixedBlock<ValueType, bs, bs>;
    const auto vals =
        reinterpret_cast<const Blk_t *>(sysmat->get_const_values());
    const auto vals_l =
        reinterpret_cast<const Blk_t *>(l_factor->get_const_values());
    const auto vals_ut =
        reinterpret_cast<const Blk_t *>(u_factor_t->get_const_values());

    Array<ValueType> n_l_values(exec, l_factor->get_num_stored_elements());
    Array<ValueType> n_u_values(exec, u_factor_t->get_num_stored_elements());
    Blk_t *const n_vals_l = reinterpret_cast<Blk_t *>(n_l_values.get_data());
    Blk_t *const n_vals_ut = reinterpret_cast<Blk_t *>(n_u_values.get_data());

#pragma omp parallel for default(shared)
    for (IndexType inz = 0; inz < l_factor->get_num_stored_elements(); inz++) {
        n_l_values.get_data()[inz] = l_factor->get_const_values()[inz];
    }
#pragma omp parallel for default(shared)
    for (IndexType inz = 0; inz < u_factor_t->get_num_stored_elements();
         inz++) {
        n_u_values.get_data()[inz] = u_factor_t->get_const_values()[inz];
    }

    for (int sweep = 0; sweep < iters; ++sweep) {
        bilu_kernel<bs, true>(nbrows, row_ptrs, col_idxs,
                              sysmat->get_const_values(), row_ptrs_l,
                              col_idxs_l, l_factor->get_values(), col_ptrs_u,
                              row_idxs_u, u_factor_t->get_values(),
                              n_l_values.get_data(), n_u_values.get_data());

#pragma omp parallel for default(shared)
        for (IndexType inz = 0; inz < l_factor->get_num_stored_elements();
             inz++) {
            l_factor->get_values()[inz] = n_l_values.get_const_data()[inz];
        }
#pragma omp parallel for default(shared)
        for (IndexType inz = 0; inz < u_factor_t->get_num_stored_elements();
             inz++) {
            u_factor_t->get_values()[inz] = n_u_values.get_const_data()[inz];
        }
    }
}

template <typename ValueType, typename IndexType>
void compute_bilu_factors_jacobi(
    const std::shared_ptr<const OmpExecutor> exec, const int iters,
    const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
    matrix::Fbcsr<ValueType, IndexType> *const lfactor,
    matrix::Fbcsr<ValueType, IndexType> *const ufactor_t)
{
    const int bs = sysmat->get_block_size();
    GKO_ASSERT(bs == lfactor->get_block_size());
    GKO_ASSERT(bs == ufactor_t->get_block_size());

    if (bs == 2) {
        compute_bilu_jac_impl<2, ValueType, IndexType>(exec, iters, sysmat,
                                                       lfactor, ufactor_t);
    } else if (bs == 3) {
        compute_bilu_jac_impl<3, ValueType, IndexType>(exec, iters, sysmat,
                                                       lfactor, ufactor_t);
    } else if (bs == 4) {
        compute_bilu_jac_impl<4, ValueType, IndexType>(exec, iters, sysmat,
                                                       lfactor, ufactor_t);
    } else if (bs == 7) {
        compute_bilu_jac_impl<7, ValueType, IndexType>(exec, iters, sysmat,
                                                       lfactor, ufactor_t);
    } else {
        throw NotSupported(__FILE__, __LINE__, __func__,
                           " block size = " + std::to_string(bs));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COMPUTE_BILU_FACTORS_FBCSR_JACOBI);


}  // namespace par_bilu_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
