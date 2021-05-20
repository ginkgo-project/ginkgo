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

#include "core/components/validation_helpers.hpp"
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/matrix_data.hpp>

#include <algorithm>
#include <cmath>

namespace gko {
namespace validate {

template <class ValueType, class IndexType>
bool is_symmetric_impl(const LinOp *matrix, const float tolerance)
{
    matrix_data<ValueType, IndexType> data{};
    dynamic_cast<const WritableToMatrixData<ValueType, IndexType> *>(matrix)
        ->write(data);

    matrix_data<ValueType, IndexType> data_t{data};
    for (auto &nonzeros : data.nonzeros) {
        std::swap(nonzeros.row, nonzeros.column);
    }

    data.ensure_row_major_order();
    data_t.ensure_row_major_order();

    return std::equal(
        data.nonzeros.begin(), data.nonzeros.end(), data_t.nonzeros.begin(),
        [tolerance](const auto v1, const auto v2) {
            return std::abs((v1.value - v2.value) / v1.value) < tolerance;
        });
}

#define GKO_CALL_AND_RETURN_IF_CASTABLE(T1, T2, func, matrix, ...)    \
    if (dynamic_cast<const WritableToMatrixData<T1, T2> *>(matrix)) { \
        return func<T1, T2>(matrix, ##__VA_ARGS__);                   \
    }

bool is_symmetric(const LinOp *matrix, const float tolerance)
{
    GKO_CALL_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_CALL_AND_RETURN_IF_CASTABLE,
                                           is_symmetric_impl, matrix, tolerance)
    return false;
}

template <class ValueType, class IndexType>
bool has_non_zero_diagonal_impl(const LinOp *matrix)
{
    matrix_data<ValueType, IndexType> data{};
    dynamic_cast<const WritableToMatrixData<ValueType, IndexType> *>(matrix)
        ->write(data);

    for (auto &nonzero : data.nonzeros) {
        if (nonzero.row == nonzero.column && std::abs(nonzero.value) == 0)
            return false;
    }

    return true;
}

bool has_non_zero_diagonal(const LinOp *matrix)
{
    GKO_CALL_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_CALL_AND_RETURN_IF_CASTABLE,
                                           has_non_zero_diagonal_impl, matrix)
    return false;
}
#undef GKO_CALL_AND_RETURN_IF_CASTABLE

template <typename IndexType>
bool has_unique_idxs(const IndexType *idxs, const size_type num_entries)
{
    for (size_type i = 1; i < num_entries; ++i) {
        if (idxs[i - 1] == idxs[i]) {
            return false;
        }
    }
    return true;
};

#define GKO_DECLARE_HAS_UNIQUE_IDXS(IndexType) \
    bool has_unique_idxs(const IndexType *idxs, const size_type num_entries)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_HAS_UNIQUE_IDXS);

template <typename IndexType>
bool is_row_ordered(const IndexType *row_ptrs, const size_type num_entries)
{
    for (size_type i = 1; i < num_entries; ++i) {
        if (row_ptrs[i - 1] > row_ptrs[i]) {
            return false;
        }
    }
    return true;
}

#define GKO_DECLARE_IS_ROW_ORDERED(IndexType) \
    bool is_row_ordered(const IndexType *row_ptrs, const size_type num_entries)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_ROW_ORDERED);

template <typename IndexType>
bool is_within_bounds(const IndexType *idxs, const size_type num_entries,
                      const IndexType lower_bound, const IndexType upper_bound)
{
    for (size_type i = 0; i < num_entries; ++i) {
        if (lower_bound > idxs[i] || idxs[i] >= upper_bound) {
            return false;
        }
    }
    return true;
}

#define GKO_DECLARE_IS_WITHIN_BOUNDS(IndexType)                               \
    bool is_within_bounds(const IndexType *idxs, const size_type num_entries, \
                          const IndexType lower_bound,                        \
                          const IndexType upper_bound)

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_WITHIN_BOUNDS);

template <typename ValueType>
bool is_finite(const ValueType *values, const size_type num_entries)
{
    for (size_type i = 0; i < num_entries; ++i) {
        if (!std::isfinite(std::abs(values[i]))) {
            return false;
        }
    }
    return true;
}

#define GKO_DECLARE_IS_FINITE(ValueType) \
    bool is_finite(const ValueType *values, const size_type num_entries)

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IS_FINITE);


}  // namespace validate

}  // namespace gko
