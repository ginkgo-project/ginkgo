// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"

#include <algorithm>
#include <vector>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace isai {


constexpr size_type default_rowwise_workspace_bytes = size_type{1} << 30;


template <typename ValueType, typename IndexType>
void generate_spd_inverse_rowwise(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<ValueType, IndexType> inverse, size_type workspace_bytes)
{
    const auto num_rows = inverse.size[0];
    if (num_rows == 0) {
        return;
    }
    // offsets of each row's packed k (k + 1) / 2 workspace
    array<int64> offsets{exec, num_rows + 1};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto num_rows, auto row_ptrs, auto offsets) {
            const int64 size =
                row < num_rows ? row_ptrs[row + 1] - row_ptrs[row] : 0;
            offsets[row] = size * (size + 1) / 2;
        },
        num_rows + 1, static_cast<int64>(num_rows), inverse.row_ptrs,
        offsets.get_data());
    components::prefix_sum_nonnegative(exec, offsets.get_data(), num_rows + 1);
    const array<int64> host_offsets{exec->get_master(), offsets};
    const auto h_offsets = host_offsets.get_const_data();
    const auto budget = static_cast<int64>(std::max<size_type>(
        1, (workspace_bytes == 0 ? default_rowwise_workspace_bytes
                                 : workspace_bytes) /
               sizeof(ValueType)));
    // chunks of rows whose workspace fits the budget, at least one row each
    std::vector<size_type> chunk_starts{0};
    int64 max_chunk = 0;
    for (size_type row = 1; row <= num_rows; ++row) {
        const auto start = chunk_starts.back();
        if (row == num_rows || h_offsets[row + 1] - h_offsets[start] > budget) {
            max_chunk = std::max(max_chunk, h_offsets[row] - h_offsets[start]);
            chunk_starts.push_back(row);
        }
    }
    array<ValueType> workspace{exec, static_cast<size_type>(max_chunk)};
    for (size_type chunk = 0; chunk + 1 < chunk_starts.size(); ++chunk) {
        const auto start = chunk_starts[chunk];
        run_kernel(
            exec,
            [] GKO_KERNEL(auto i, auto start, auto offsets, auto m_row_ptrs,
                          auto m_cols, auto m_vals, auto i_row_ptrs,
                          auto i_cols, auto i_vals, auto workspace) {
                const auto row = start + i;
                const auto begin = i_row_ptrs[row];
                const auto size = i_row_ptrs[row + 1] - begin;
                const auto pattern = i_cols + begin;
                const auto out = i_vals + begin;
                const auto m = workspace + (offsets[row] - offsets[start]);
                auto packed = [](int64 a, int64 b) {
                    return a * (a + 1) / 2 + b;
                };
                // M(a, b) = conj(A(S_a, S_b)) for b <= a
                for (int64 a = 0; a < size; ++a) {
                    for (int64 b = 0; b <= a; ++b) {
                        m[packed(a, b)] = zero(m[0]);
                    }
                    auto nz = m_row_ptrs[pattern[a]];
                    const auto end = m_row_ptrs[pattern[a] + 1];
                    int64 b = 0;
                    while (nz < end && b <= a) {
                        const auto col = m_cols[nz];
                        if (col == pattern[b]) {
                            m[packed(a, b)] = conj(m_vals[nz]);
                        }
                        nz += col <= pattern[b];
                        b += col >= pattern[b];
                    }
                }
                // M = C C^H in place, then C^H z = e_k into out
                bool valid = true;
                for (int64 j = 0; j < size && valid; ++j) {
                    const auto diag = real(m[packed(j, j)]);
                    valid = diag > zero(diag) && is_finite(diag);
                    if (valid) {
                        const auto c = sqrt(diag);
                        m[packed(j, j)] = c;
                        for (int64 r = j + 1; r < size; ++r) {
                            m[packed(r, j)] /= c;
                        }
                        for (int64 r = j + 1; r < size; ++r) {
                            for (int64 l = j + 1; l <= r; ++l) {
                                m[packed(r, l)] -=
                                    m[packed(r, j)] * conj(m[packed(l, j)]);
                            }
                        }
                    }
                }
                for (int64 r = size - 1; r >= 0 && valid; --r) {
                    auto sum = r == size - 1 ? one(m[0]) : zero(m[0]);
                    for (int64 l = r + 1; l < size; ++l) {
                        sum -= conj(m[packed(l, r)]) * out[l];
                    }
                    out[r] = sum / m[packed(r, r)];
                    valid = is_finite(out[r]);
                }
                if (!valid) {
                    for (int64 a = 0; a < size; ++a) {
                        out[a] = pattern[a] == row ? one(m[0]) : zero(m[0]);
                    }
                }
            },
            chunk_starts[chunk + 1] - start, static_cast<int64>(start),
            offsets.get_const_data(), input.row_ptrs, input.col_idxs,
            input.values, inverse.row_ptrs, inverse.col_idxs, inverse.values,
            workspace.get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_SPD_INVERSE_ROWWISE_KERNEL);


}  // namespace isai
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
