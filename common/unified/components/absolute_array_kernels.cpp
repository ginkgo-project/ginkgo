// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/absolute_array_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType>
void inplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                            ValueType* data, size_type n)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto data) { data[idx] = abs(data[idx]); }, n,
        data);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_ARRAY_KERNEL);


template <typename ValueType>
void outplace_absolute_array(std::shared_ptr<const DefaultExecutor> exec,
                             const ValueType* in, size_type n,
                             remove_complex<ValueType>* out)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto in, auto out) { out[idx] = abs(in[idx]); },
        n, in, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_ARRAY_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
