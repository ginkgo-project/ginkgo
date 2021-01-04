/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/preconditioner/isai_kernels.hpp"


#include <algorithm>
#include <memory>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Isai preconditioner namespace.
 *
 * @ingroup isai
 */
namespace isai {


template <typename IndexType, typename Callback>
void forall_matching(const IndexType *fst, IndexType fst_size,
                     const IndexType *snd, IndexType snd_size, Callback cb)
{
    IndexType fst_idx{};
    IndexType snd_idx{};
    while (fst_idx < fst_size && snd_idx < snd_size) {
        const auto fst_val = fst[fst_idx];
        const auto snd_val = snd[snd_idx];
        if (fst_val == snd_val) {
            cb(fst_val, fst_idx, snd_idx);
        }
        // advance the smaller entrie(s)
        fst_idx += (fst_val <= snd_val);
        snd_idx += (fst_val >= snd_val);
    }
}


template <typename ValueType, typename IndexType, typename Callable>
void generic_generate(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *mtx,
                      matrix::Csr<ValueType, IndexType> *inverse_mtx,
                      IndexType *excess_rhs_ptrs, IndexType *excess_nz_ptrs,
                      Callable trs_solve)
{
    /*
    Consider: aiM := inverse_mtx; M := mtx
    I := Identity matrix
    e(i) := unit vector i (containing all zeros except for row i, which is one)
    S := Sparsity pattern of the desired aiM
    S(i) := Sparsity pattern of row i of aiM (Set of non-zero columns)
    D(i) := M[S(i), S(i)]
    aiM := approximate inverse of M

    Target: Solving (aiM * M = I)_{S} (aiM * M = I for the sparsity pattern S)
    aiM[i, :] * D(i) = e(i)^T
    <=> D(i)^T * aiM[i, :]^T = e(i)   =^ Triangular system (Trs)
    Solve Trs, fill in aiM row by row (coalesced access)
    */
    const auto num_rows = mtx->get_size()[0];
    const auto m_row_ptrs = mtx->get_const_row_ptrs();
    const auto m_cols = mtx->get_const_col_idxs();
    const auto m_vals = mtx->get_const_values();
    const auto i_row_ptrs = inverse_mtx->get_const_row_ptrs();
    const auto i_cols = inverse_mtx->get_const_col_idxs();
    auto i_vals = inverse_mtx->get_values();

    auto num_threads = static_cast<size_type>(omp_get_max_threads());
    // RHS for local trisystem
    gko::Array<ValueType> rhs_array{exec, row_size_limit * num_threads};
    // memory for dense trisystem
    gko::Array<ValueType> trisystem_array{
        exec, row_size_limit * row_size_limit * num_threads};

#pragma omp parallel
    {
        auto thread_num = static_cast<size_type>(omp_get_thread_num());

        auto rhs = rhs_array.get_data() + thread_num * row_size_limit;
        auto trisystem_ptr = trisystem_array.get_data() +
                             thread_num * row_size_limit * row_size_limit;

#pragma omp for
        for (size_type row = 0; row < num_rows; ++row) {
            const auto i_begin = i_row_ptrs[row];
            const auto i_size = i_row_ptrs[row + 1] - i_begin;

            if (i_size <= row_size_limit) {
                // short rows: treat directly as dense system
                excess_rhs_ptrs[row] = 0;
                excess_nz_ptrs[row] = 0;
                auto trisystem = range<accessor::row_major<ValueType, 2>>(
                    trisystem_ptr, static_cast<size_type>(i_size),
                    static_cast<size_type>(i_size),
                    static_cast<size_type>(i_size));
                std::fill_n(trisystem_ptr, i_size * i_size, zero<ValueType>());

                for (size_type i = 0; i < i_size; ++i) {
                    const auto col = i_cols[i_begin + i];
                    const auto m_begin = m_row_ptrs[col];
                    const auto m_size = m_row_ptrs[col + 1] - m_begin;
                    forall_matching(
                        m_cols + m_begin, m_size, i_cols + i_begin, i_size,
                        [&](IndexType, IndexType m_idx, IndexType i_idx) {
                            trisystem(i, i_idx) = m_vals[m_idx + m_begin];
                        });
                }

                // solve dense triangular system
                trs_solve(trisystem, rhs);

                // write triangular solution to inverse
                for (size_type i = 0; i < i_size; ++i) {
                    const auto new_val = rhs[i];
                    const auto idx = i_begin + i;
                    // check for non-finite elements which should not be copied
                    // over
                    if (is_finite(new_val)) {
                        i_vals[idx] = new_val;
                    } else {
                        // ensure the preconditioner does not prevent
                        // convergence
                        i_vals[idx] = i_cols[idx] == row ? one<ValueType>()
                                                         : zero<ValueType>();
                    }
                }
            } else {
                // count non-zeros and dimension in the excess system
                IndexType count{};
                for (size_type i = 0; i < i_size; ++i) {
                    const auto col = i_cols[i_begin + i];
                    const auto m_begin = m_row_ptrs[col];
                    const auto m_size = m_row_ptrs[col + 1] - m_begin;
                    forall_matching(
                        m_cols + m_begin, m_size, i_cols + i_begin, i_size,
                        [&](IndexType, IndexType, IndexType) { ++count; });
                }
                excess_rhs_ptrs[row] = i_size;
                excess_nz_ptrs[row] = count;
            }
        }
    }
    components::prefix_sum(exec, excess_rhs_ptrs, num_rows + 1);
    components::prefix_sum(exec, excess_nz_ptrs, num_rows + 1);
}


template <typename ValueType, typename IndexType>
void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *mtx,
                          matrix::Csr<ValueType, IndexType> *inverse_mtx,
                          IndexType *excess_rhs_ptrs, IndexType *excess_nz_ptrs,
                          bool lower)
{
    auto trs_solve =
        [lower](const range<accessor::row_major<ValueType, 2>> trisystem,
                ValueType *rhs) {
            const IndexType size = trisystem.length(0);
            if (size <= 0) {
                return;
            }
            // RHS is the identity: zero everywhere except for the diagonal
            // entry
            std::fill_n(rhs, size, zero<ValueType>());
            rhs[lower ? size - 1 : 0] = one<ValueType>();

            // solve transposed triangular system
            if (lower) {
                for (auto col = size - 1; col >= 0; --col) {
                    const auto diag = trisystem(col, col);
                    const auto bot = rhs[col] / diag;
                    rhs[col] = bot;
                    // do a backwards substitution
                    for (auto row = col - 1; row >= 0; --row) {
                        rhs[row] -= bot * trisystem(col, row);
                    }
                }
            } else {
                for (IndexType col = 0; col < size; ++col) {
                    const auto diag = trisystem(col, col);
                    const auto top = rhs[col] / diag;
                    rhs[col] = top;
                    // do a forward substitution
                    for (auto row = col + 1; row < size; ++row) {
                        rhs[row] -= top * trisystem(col, row);
                    }
                }
            }
        };

    generic_generate(exec, mtx, inverse_mtx, excess_rhs_ptrs, excess_nz_ptrs,
                     trs_solve);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(std::shared_ptr<const DefaultExecutor>,
                            const matrix::Csr<ValueType, IndexType> *input,
                            const matrix::Csr<ValueType, IndexType> *inverse,
                            const IndexType *excess_rhs_ptrs,
                            const IndexType *excess_nz_ptrs,
                            matrix::Csr<ValueType, IndexType> *excess_system,
                            matrix::Dense<ValueType> *excess_rhs)
{
    const auto num_rows = input->get_size()[0];
    const auto m_row_ptrs = input->get_const_row_ptrs();
    const auto m_cols = input->get_const_col_idxs();
    const auto m_vals = input->get_const_values();
    const auto i_row_ptrs = inverse->get_const_row_ptrs();
    const auto i_cols = inverse->get_const_col_idxs();
    const auto e_dim = excess_rhs->get_size()[0];
    auto e_row_ptrs = excess_system->get_row_ptrs();
    auto e_cols = excess_system->get_col_idxs();
    auto e_vals = excess_system->get_values();
    auto e_rhs = excess_rhs->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        const auto i_begin = i_row_ptrs[row];
        const auto i_size = i_row_ptrs[row + 1] - i_begin;
        // first row index of the sparse block in the excess system
        auto e_begin = excess_rhs_ptrs[row];
        // first non-zero index in the sparse block
        auto e_nz = excess_nz_ptrs[row];

        if (i_size > row_size_limit) {
            // count non-zeros and dimension in the excess system
            for (size_type i = 0; i < i_size; ++i) {
                // current row in the excess system
                const auto e_row = e_begin + i;
                const auto col = i_cols[i_begin + i];
                const auto m_begin = m_row_ptrs[col];
                const auto m_size = m_row_ptrs[col + 1] - m_begin;
                // store row pointers: one row per non-zero of inverse row
                e_row_ptrs[e_row] = e_nz;
                // build right-hand side: identity row
                e_rhs[e_row] =
                    row == col ? one<ValueType>() : zero<ValueType>();
                // build sparse block
                forall_matching(
                    m_cols + m_begin, m_size, i_cols + i_begin, i_size,
                    [&](IndexType, IndexType m_idx, IndexType i_idx) {
                        // trisystem(i, i_idx) = m_vals[m_idx + m_begin]
                        // just in sparse
                        e_cols[e_nz] = i_idx + e_begin;
                        e_vals[e_nz] = m_vals[m_idx + m_begin];
                        ++e_nz;
                    });
            }
        }
    }
    e_row_ptrs[e_dim] = excess_nz_ptrs[num_rows];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(std::shared_ptr<const DefaultExecutor>,
                             const IndexType *excess_block_ptrs,
                             const matrix::Dense<ValueType> *excess_solution,
                             matrix::Csr<ValueType, IndexType> *inverse)
{
    const auto num_rows = inverse->get_size()[0];
    auto excess_values = excess_solution->get_const_values();
    auto values = inverse->get_values();
    auto row_ptrs = inverse->get_const_row_ptrs();
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        const auto excess_begin = excess_values + excess_block_ptrs[row];
        const auto excess_end = excess_values + excess_block_ptrs[row + 1];
        auto values_begin = values + row_ptrs[row];
        std::copy(excess_begin, excess_end, values_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


}  // namespace isai
}  // namespace omp
}  // namespace kernels
}  // namespace gko
