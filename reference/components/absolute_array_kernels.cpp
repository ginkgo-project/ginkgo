// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/absolute_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType>
void inplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                            ValueType* data, size_type n)
{
    for (size_type i = 0; i < n; i++) {
        data[i] = abs(data[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_ARRAY_KERNEL);


template <typename ValueType>
void outplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                             const ValueType* in, size_type n,
                             remove_complex<ValueType>* out)
{
    for (size_type i = 0; i < n; i++) {
        out[i] = abs(in[i]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_ARRAY_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
