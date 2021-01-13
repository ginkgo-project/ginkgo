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


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The parallel iterative block ILU factorization namespace.
 *
 * @ingroup factor
 */
namespace par_bilu_factorization {


template <typename ValueType, typename IndexType, int bs>
static void compute_bilu_impl(
    const std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor)
{
    const IndexType *const col_idxs = sysmat->get_const_col_idxs();
    const IndexType *const row_ptrs = sysmat->get_const_row_ptrs();
    const IndexType *const row_ptrs_l = l_factor->get_const_row_ptrs();
    const IndexType *const col_ptrs_u = u_factor->get_const_row_ptrs();
    const IndexType *const col_idxs_l = l_factor->get_const_col_idxs();
    const IndexType *const row_idxs_u = u_factor->get_const_col_idxs();

    using Blk_t = blockutils::FixedBlock<ValueType, bs, bs>;
    const auto vals = reinterpret_cast<const Blk_t *>(sysmat->get_values());
    const auto vals_l = reinterpret_cast<Blk_t *>(l_factor->get_values());
    const auto vals_u = reinterpret_cast<Blk_t *>(u_factor->get_values());

    for (IndexType ibrow = 0;
         ibrow < static_cast<IndexType>(sysmat->get_num_block_rows());
         ++ibrow) {
        for (IndexType ibz = row_ptrs[ibrow]; ibz < row_ptrs[ibrow + 1];
             ibz++) {
            // const auto row = row_idxs[el];
            const auto jbcol = col_idxs[ibz];
            // const auto val = vals[el];
            auto lbz = row_ptrs_l[ibrow];
            auto ubz = col_ptrs_u[jbcol];

            Blk_t sum;
            Blk_t last_op;
            ValueType *const sumarr = &sum(0, 0);
            ValueType *const lastarr = &last_op(0, 0);
            const ValueType *const valarr = &vals[ibz](0, 0);
            for (int i = 0; i < bs * bs; i++) {
                sumarr[i] = valarr[i];
                lastarr[i] = zero<ValueType>;
            }

            // Calculate: sum = system_matrix(row, col) -
            //   dot(l_factor(row, :), u_factor(:, col))
            while (lbz < row_ptrs_l[ibrow + 1] && ubz < col_ptrs_u[jbcol + 1]) {
                const auto bcol_l = col_idxs_l[lbz];
                const auto bcol_u = row_idxs_u[ubz];
                if (bcol_l == bcol_u) {
                    // last_op = vals_l[row_l] * vals_u[row_u];
                    // sum -= last_op;
                    for (int i = 0; i < bs; i++)
                        for (int j = 0; i < bs; j++) {
                            last_op(i, j) = zero<ValueType>;
                            for (int k = 0; k < bs; k++)
                                last_op(i, j) +=
                                    vals_l[row_l](i, k) * vals_u[row_u](k, j);
                            sum(i, j) -= last_op(i, j);
                        }
                }

                if (bcol_l <= bcol_u) ++lbz;

                if (bcol_u <= bcol_l) ++ubz;
            }

            // undo the last operation
            for (int i = 0; i < bs; i++)
                for (int j = 0; i < bs; j++) sum(i, j) += last_op(i, j);

            if (ibrow > jbcol) {
                // modify entry in L
                Blk_t invU;
                for (int i = 0; i < bs; i++)
                    for (int j = 0; i < bs; j++)
                        invU(i, j) = vals_u[row_ptrs_u[col + 1] - 1](i, j);

                int perm[bs];
                for (int i = 0; i < bs; i++) perm[i] = i;

                const bool invflag = invert_block<ValueType, bs>(perm, invU);

                // auto to_write = sum / vals_u[row_ptrs_u[col + 1] - 1];
                // if (is_finite(to_write))
                // vals_l[lbz - 1] = to_write;
                for (int i = 0; i < bs; i++)
                    for (int j = 0; j < bs; j++) {
                        vals_l[lbz - 1](i, j) = 0;
                        for (int k = 0; k < bs; k++)
                            vals_l[lbz - 1](i, j) += sum(i, j) * invU(i, j);
                    }
            } else {
                // modify entry in U
                auto to_write = sum;
                if (is_finite(to_write)) {
                    vals_u[ubz - 1] = to_write;
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void compute_bilu(const std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Fbcsr<ValueType, IndexType> *const sysmat,
                  matrix::Fbcsr<ValueType, IndexType> *const lfactor,
                  matrix::Fbcsr<ValueType, IndexType> *const ufactor)
{
    const int bs = sysmat->get_block_size();
    GKO_ASSERT(bs == lfactor->get_block_size());
    GKO_ASSERT(bs == ufactor->get_block_size());

    if (bs == 2)
        compute_bilu_impl<ValueType, IndexType, 2>(exec, sysmat, lfactor,
                                                   ufactor);
    else if (bs == 4)
        compute_bilu_impl<ValueType, IndexType, 4>(exec, sysmat, lfactor,
                                                   ufactor);
    else
        throw NotSupported(__FILE__, __LINE__, __func__,
                           " block size = " + std::to_string(bs));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BILU_COMPUTE_BLU_KERNEL);


}  // namespace par_bilu_factorization
}  // namespace reference
}  // namespace kernels
}  // namespace gko
