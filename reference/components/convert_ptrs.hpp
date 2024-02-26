// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <algorithm>
#include <numeric>


#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
namespace csr {


template <typename IndexType>
inline void convert_ptrs_to_idxs(const IndexType* ptrs, size_type num_rows,
                                 IndexType* idxs)
{
    size_type ind = 0;

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type i = ptrs[row]; i < static_cast<size_type>(ptrs[row + 1]);
             ++i) {
            idxs[ind++] = row;
        }
    }
}


}  // namespace csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
