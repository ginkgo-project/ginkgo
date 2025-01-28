// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "apply.hpp"

#include <ginkgo/core/base/math.hpp>


using gko::half;


template <typename ValueType>
void advanced_apply(gko::size_type id, gko::dim<2> size, const ValueType alpha,
                    const ValueType* b, const ValueType beta, ValueType* x,
                    void* payload)
{
    auto num_batches = *reinterpret_cast<gko::size_type*>(payload);
    for (gko::size_type row = 0; row < size[0]; ++row) {
        ValueType acc{};

        if (row > 0) {
            acc += -gko::one<ValueType>() * b[row - 1];
        }
        acc +=
            (static_cast<ValueType>(2.0) +
             static_cast<ValueType>(id) / static_cast<ValueType>(num_batches)) *
            b[row];
        if (row < size[0] - 1) {
            acc += -gko::one<ValueType>() * b[row + 1];
        }
        x[row] = alpha * acc + beta * x[row];
    }
}


template <typename ValueType>
void advanced_apply_generic(gko::size_type id, gko::dim<2> size,
                            const void* alpha, const void* b, const void* beta,
                            void* x, void* payload)
{
    advanced_apply(id, size, *reinterpret_cast<const ValueType*>(alpha),
                   reinterpret_cast<const ValueType*>(b),
                   *reinterpret_cast<const ValueType*>(beta),
                   reinterpret_cast<ValueType*>(x), payload);
}

#define GKO_DECLARE_ADVANCED_APPLY(_vtype)                                     \
    void advanced_apply_generic<_vtype>(                                       \
        gko::size_type id, gko::dim<2> size, const void* alpha, const void* b, \
        const void* beta, void* x, void* payload)

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ADVANCED_APPLY);


template <typename ValueType>
void simple_apply_generic(gko::size_type id, gko::dim<2> size, const void* b,
                          void* x, void* payload)
{
    advanced_apply(
        id, size, gko::one<ValueType>(), reinterpret_cast<const ValueType*>(b),
        gko::zero<ValueType>(), reinterpret_cast<ValueType*>(x), payload);
}

#define GKO_DECLARE_SIMPLE_APPLY(_vtype)                                   \
    void simple_apply_generic<_vtype>(gko::size_type id, gko::dim<2> size, \
                                      const void* b, void* x, void* payload)

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_SIMPLE_APPLY);
