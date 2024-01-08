// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Isai preconditioner namespace.
 *
 * @ingroup isai
 */
namespace isai {


template <typename IndexType, typename Callback>
void forall_matching(const IndexType* fst, IndexType fst_size,
                     const IndexType* snd, IndexType snd_size, Callback cb)
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
                      const matrix::Csr<ValueType, IndexType>* mtx,
                      matrix::Csr<ValueType, IndexType>* inverse_mtx,
                      IndexType* excess_rhs_ptrs, IndexType* excess_nz_ptrs,
                      Callable direct_solve, bool tri)
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
    // RHS for local dense system
    gko::array<ValueType> rhs_array{exec, row_size_limit};
    auto rhs = rhs_array.get_data();
    // memory for dense dense system
    gko::array<ValueType> dense_system_array{exec,
                                             row_size_limit * row_size_limit};
    auto dense_system_ptr = dense_system_array.get_data();
    // stores the next free index in the excess rhs/solution
    IndexType excess_rhs_begin{};
    // stores the next free non-zero index in the excess system
    IndexType excess_nz_begin{};

    for (size_type row = 0; row < num_rows; ++row) {
        const auto i_begin = i_row_ptrs[row];
        const auto i_size = i_row_ptrs[row + 1] - i_begin;
        excess_rhs_ptrs[row] = excess_rhs_begin;
        excess_nz_ptrs[row] = excess_nz_begin;

        if (i_size <= row_size_limit) {
            // short rows: treat directly as dense system
            // we need this ugly workaround to get rid of a few
            // warnings and compilation issues
            auto dense_system = range<accessor::row_major<ValueType, 2>>(
                dense_system_ptr, static_cast<size_type>(i_size),
                static_cast<size_type>(i_size), static_cast<size_type>(i_size));
            std::fill_n(dense_system_ptr, i_size * i_size, zero<ValueType>());
            // For general ISAI, the index of the one in the rhs depends on
            // the number of nonzeros in the lower half of the matrix of the
            // according row.
            IndexType rhs_one_idx = zero<IndexType>();

            for (size_type i = 0; i < i_size; ++i) {
                const auto col = i_cols[i_begin + i];
                const auto m_begin = m_row_ptrs[col];
                const auto m_size = m_row_ptrs[col + 1] - m_begin;
                // Loop over all matches that are within the sparsity pattern of
                // the original matrix instead of the larger pattern of the
                // inverse because any match outside the narrower sparsity
                // pattern results in a zero entry
                forall_matching(
                    m_cols + m_begin, m_size, i_cols + i_begin, i_size,
                    [&](IndexType, IndexType m_idx, IndexType i_idx) {
                        if (tri) {
                            dense_system(i, i_idx) = m_vals[m_idx + m_begin];
                        } else {
                            dense_system(i_idx, i) = m_vals[m_idx + m_begin];
                        }
                    });
                const auto i_transposed_row_begin = i_row_ptrs[col];
                const auto i_transposed_row_size =
                    i_row_ptrs[col + 1] - i_transposed_row_begin;
                // Loop over all matches that are within the sparsity pattern of
                // the inverse
                forall_matching(
                    i_cols + i_transposed_row_begin, i_transposed_row_size,
                    i_cols + i_begin, i_size,
                    [&](IndexType, IndexType m_idx, IndexType i_idx) {
                        if (i_cols[m_idx + i_transposed_row_begin] < row &&
                            col == row) {
                            rhs_one_idx++;
                        }
                    });
            }

            // solve dense triangular system
            direct_solve(dense_system, rhs, rhs_one_idx);

            // write triangular solution to inverse
            for (size_type i = 0; i < i_size; ++i) {
                const auto new_val = rhs[i];
                const auto idx = i_begin + i;
                // check for non-finite elements which should not be copied over
                if (is_finite(new_val)) {
                    i_vals[idx] = new_val;
                } else {
                    // ensure the preconditioner does not prevent convergence
                    i_vals[idx] = i_cols[idx] == row ? one<ValueType>()
                                                     : zero<ValueType>();
                }
            }
        } else {
            // count non-zeros and dimension in the excess system
            for (size_type i = 0; i < i_size; ++i) {
                const auto col = i_cols[i_begin + i];
                const auto m_begin = m_row_ptrs[col];
                const auto m_size = m_row_ptrs[col + 1] - m_begin;
                forall_matching(m_cols + m_begin, m_size, i_cols + i_begin,
                                i_size, [&](IndexType, IndexType, IndexType) {
                                    ++excess_nz_begin;
                                });
                ++excess_rhs_begin;
            }
        }
    }
    excess_rhs_ptrs[num_rows] = excess_rhs_begin;
    excess_nz_ptrs[num_rows] = excess_nz_begin;
}


template <typename ValueType, typename IndexType>
void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType>* mtx,
                          matrix::Csr<ValueType, IndexType>* inverse_mtx,
                          IndexType* excess_rhs_ptrs, IndexType* excess_nz_ptrs,
                          bool lower)
{
    auto trs_solve =
        [lower](const range<accessor::row_major<ValueType, 2>> trisystem,
                ValueType* rhs, const IndexType) {
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
                     trs_solve, true);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
inline IndexType choose_pivot(IndexType block_size, const ValueType* block,
                              size_type stride)
{
    IndexType cp = 0;
    for (IndexType i = 1; i < block_size; ++i) {
        if (abs(block[cp * stride]) < abs(block[i * stride])) {
            cp = i;
        }
    }
    return cp;
}


template <typename ValueType, typename IndexType>
inline void swap_rows(IndexType row1, IndexType row2, IndexType block_size,
                      ValueType* block, size_type stride)
{
    using std::swap;
    for (IndexType i = 0; i < block_size; ++i) {
        swap(block[row1 * stride + i], block[row2 * stride + i]);
    }
}


template <typename ValueType, typename IndexType>
void generate_general_inverse(std::shared_ptr<const DefaultExecutor> exec,
                              const matrix::Csr<ValueType, IndexType>* mtx,
                              matrix::Csr<ValueType, IndexType>* inverse_mtx,
                              IndexType* excess_rhs_ptrs,
                              IndexType* excess_nz_ptrs, bool spd)
{
    using std::swap;
    auto general_solve = [spd](const range<accessor::row_major<ValueType, 2>>
                                   transposed_system_range,
                               ValueType* rhs, const IndexType rhs_one_idx) {
        const IndexType size = transposed_system_range.length(0);
        if (size <= 0) {
            return;
        }
        // RHS is the identity: zero everywhere except for the diagonal
        // entry
        std::fill_n(rhs, size, zero<ValueType>());
        rhs[rhs_one_idx] = one<ValueType>();

        auto transposed_system = transposed_system_range.get_accessor().data;

        // solve transposed system
        for (IndexType col = 0; col < size; col++) {
            const auto row =
                choose_pivot(size - col, transposed_system + col * (size + 1),
                             size) +
                col;
            swap_rows(col, row, size, transposed_system, size);
            swap(rhs[row], rhs[col]);

            const auto d = transposed_system[col * size + col];

            for (IndexType i = 0; i < size; ++i) {
                transposed_system[i * size + col] /= -d;
            }

            transposed_system[col * size + col] = zero<ValueType>();
            const auto rhs_key_val = rhs[col];
            for (IndexType i = 0; i < size; ++i) {
                const auto scal = transposed_system[i * size + col];
                for (IndexType j = 0; j < size; ++j) {
                    transposed_system[i * size + j] +=
                        scal * transposed_system[col * size + j];
                }
                rhs[i] += rhs_key_val * scal;
            }
            for (IndexType j = 0; j < size; ++j) {
                transposed_system[col * size + j] /= d;
            }
            rhs[col] /= d;
            transposed_system[col * size + col] = one<ValueType>() / d;
        }

        if (spd) {
            const auto scal = one<ValueType>() / sqrt(rhs[size - 1]);
            for (IndexType row = 0; row < size; row++) {
                rhs[row] *= scal;
            }
        }
    };

    generic_generate(exec, mtx, inverse_mtx, excess_rhs_ptrs, excess_nz_ptrs,
                     general_solve, false);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(std::shared_ptr<const DefaultExecutor>,
                            const matrix::Csr<ValueType, IndexType>* input,
                            const matrix::Csr<ValueType, IndexType>* inverse,
                            const IndexType* excess_rhs_ptrs,
                            const IndexType* excess_nz_ptrs,
                            matrix::Csr<ValueType, IndexType>* excess_system,
                            matrix::Dense<ValueType>* excess_rhs,
                            size_type e_start, size_type e_end)
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

    for (size_type row = e_start; row < e_end; ++row) {
        const auto i_begin = i_row_ptrs[row];
        const auto i_size = i_row_ptrs[row + 1] - i_begin;
        // first row index of the sparse block in the excess system
        auto e_begin = excess_rhs_ptrs[row] - excess_rhs_ptrs[e_start];
        // first non-zero index in the sparse block
        auto e_nz = excess_nz_ptrs[row] - excess_nz_ptrs[e_start];

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
    e_row_ptrs[e_dim] = excess_nz_ptrs[e_end] - excess_nz_ptrs[e_start];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scale_excess_solution(std::shared_ptr<const DefaultExecutor>,
                           const IndexType* excess_block_ptrs,
                           matrix::Dense<ValueType>* excess_solution,
                           size_type e_start, size_type e_end)
{
    auto excess_values = excess_solution->get_values();
    IndexType block_start = 0;
    IndexType block_end = 0;
    auto offset = excess_block_ptrs[e_start];
    for (size_type row = e_start; row < e_end; ++row) {
        block_start = excess_block_ptrs[row] - offset;
        block_end = excess_block_ptrs[row + 1] - offset;
        if (block_end == block_start) {
            continue;
        }
        const ValueType scal =
            one<ValueType>() / sqrt(excess_values[block_end - 1]);
        for (size_type i = block_start; i < block_end; i++) {
            excess_values[i] *= scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(std::shared_ptr<const DefaultExecutor>,
                             const IndexType* excess_block_ptrs,
                             const matrix::Dense<ValueType>* excess_solution,
                             matrix::Csr<ValueType, IndexType>* inverse,
                             size_type e_start, size_type e_end)
{
    auto excess_values = excess_solution->get_const_values();
    auto values = inverse->get_values();
    auto row_ptrs = inverse->get_const_row_ptrs();
    auto offset = excess_block_ptrs[e_start];
    for (size_type row = e_start; row < e_end; ++row) {
        const auto excess_begin =
            excess_values + excess_block_ptrs[row] - offset;
        const auto excess_end =
            excess_values + excess_block_ptrs[row + 1] - offset;
        auto values_begin = values + row_ptrs[row];
        std::copy(excess_begin, excess_end, values_begin);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


}  // namespace isai
}  // namespace reference
}  // namespace kernels
}  // namespace gko
