// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/fill_array_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename ValueType>
void fill_array(std::shared_ptr<const DefaultExecutor> exec, ValueType* array,
                size_type n, ValueType val)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto array, auto val) { array[idx] = val; }, n,
        array, val);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
template GKO_DECLARE_FILL_ARRAY_KERNEL(bool);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint16);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint32);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint64);


template <typename ValueType>
void fill_seq_array(std::shared_ptr<const DefaultExecutor> exec,
                    ValueType* array, size_type n)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto idx, auto array) {
            if constexpr (std::is_same_v<remove_complex<ValueType>, half>) {
                // __half can not be from int64_t
                array[idx] = static_cast<long long>(idx);
            } else {
                array[idx] = idx;
            }
        },
        n, array);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
