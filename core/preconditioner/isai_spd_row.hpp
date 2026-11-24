// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_PRECONDITIONER_ISAI_SPD_ROW_HPP_
#define GKO_CORE_PRECONDITIONER_ISAI_SPD_ROW_HPP_


#include <algorithm>
#include <vector>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace isai_spd {


/**
 * Row `row` of the SPD ISAI on the host: M(a, b) = conj(A(S_a, S_b)),
 * M = C C^H, C^H z = e_k; `scratch` grows to the row's packed k (k + 1) / 2.
 */
template <typename ValueType, typename IndexType>
void generate_row(IndexType row,
                  matrix::view::csr<const ValueType, const IndexType> input,
                  matrix::view::csr<ValueType, IndexType> inverse,
                  std::vector<ValueType>& scratch)
{
    const auto begin = inverse.row_ptrs[row];
    const auto size = inverse.row_ptrs[row + 1] - begin;
    const auto pattern = inverse.col_idxs + begin;
    const auto out = inverse.values + begin;
    const auto packed_size = static_cast<size_type>(size) * (size + 1) / 2;
    if (scratch.size() < packed_size) {
        scratch.resize(packed_size);
    }
    auto m = [&](IndexType a, IndexType b) -> ValueType& {
        return scratch[static_cast<size_type>(a) * (a + 1) / 2 + b];
    };
    // M(a, b) = conj(A(S_a, S_b)) for b <= a, zero outside A's pattern
    for (IndexType a = 0; a < size; ++a) {
        std::fill_n(&m(a, 0), a + 1, zero<ValueType>());
        auto nz = input.row_ptrs[pattern[a]];
        const auto end = input.row_ptrs[pattern[a] + 1];
        IndexType b = 0;
        while (nz < end && b <= a) {
            const auto col = input.col_idxs[nz];
            if (col == pattern[b]) {
                m(a, b) = conj(input.values[nz]);
            }
            nz += col <= pattern[b];
            b += col >= pattern[b];
        }
    }
    // M = C C^H in place, then C^H z = e_k
    bool valid = true;
    for (IndexType j = 0; j < size && valid; ++j) {
        const auto diag = real(m(j, j));
        valid = diag > zero<remove_complex<ValueType>>() && is_finite(diag);
        if (valid) {
            const auto c = sqrt(diag);
            m(j, j) = c;
            for (IndexType i = j + 1; i < size; ++i) {
                m(i, j) /= c;
            }
            for (IndexType i = j + 1; i < size; ++i) {
                for (IndexType l = j + 1; l <= i; ++l) {
                    m(i, l) -= m(i, j) * conj(m(l, j));
                }
            }
        }
    }
    for (IndexType i = size - 1; i >= 0 && valid; --i) {
        auto sum = i == size - 1 ? one<ValueType>() : zero<ValueType>();
        for (IndexType l = i + 1; l < size; ++l) {
            sum -= conj(m(l, i)) * out[l];
        }
        out[i] = sum / m(i, i);
        valid = is_finite(out[i]);
    }
    if (!valid) {
        for (IndexType a = 0; a < size; ++a) {
            out[a] = pattern[a] == row ? one<ValueType>() : zero<ValueType>();
        }
    }
}


}  // namespace isai_spd
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ISAI_SPD_ROW_HPP_
