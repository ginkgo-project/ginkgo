// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/fill_array_kernels.hpp"

#include <numeric>
#include <type_traits>


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename ValueType>
void fill_array(std::shared_ptr<const DefaultExecutor> exec, ValueType* array,
                size_type n, ValueType val)
{
    std::fill_n(array, n, val);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
template GKO_DECLARE_FILL_ARRAY_KERNEL(bool);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint16);
template GKO_DECLARE_FILL_ARRAY_KERNEL(uint32);


// this is necessary because compilers use different types for uint64_t and
// size_t, namely unsigned long long and unsigned long
void fill_array_instantiation_helper()
{
    fill_array<uint64>(nullptr, nullptr, 0, 0);
}


template <typename ValueType>
void fill_seq_array(std::shared_ptr<const DefaultExecutor> exec,
                    ValueType* array, size_type n)
{
    std::iota(array, array + n, 0);
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_FILL_SEQ_ARRAY_KERNEL);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
