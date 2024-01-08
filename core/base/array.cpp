// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>


#include <ginkgo/core/base/math.hpp>


#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/precision_conversion_kernels.hpp"
#include "core/components/reduce_array_kernels.hpp"


namespace gko {
namespace conversion {
namespace {


GKO_REGISTER_OPERATION(convert, components::convert_precision);


}  // anonymous namespace
}  // namespace conversion


namespace array_kernels {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(reduce_add_array, components::reduce_add_array);


}  // anonymous namespace
}  // namespace array_kernels


namespace detail {


template <typename SourceType, typename TargetType>
void convert_data(std::shared_ptr<const Executor> exec, size_type size,
                  const SourceType* src, TargetType* dst)
{
    exec->run(conversion::make_convert(size, src, dst));
}


#define GKO_DECLARE_ARRAY_CONVERSION(From, To)                              \
    void convert_data<From, To>(std::shared_ptr<const Executor>, size_type, \
                                const From*, To*)

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_ARRAY_CONVERSION);


}  // namespace detail


template <typename ValueType>
void array<ValueType>::fill(const ValueType value)
{
    this->get_executor()->run(array_kernels::make_fill_array(
        this->get_data(), this->get_size(), value));
}


template <typename ValueType>
void reduce_add(const array<ValueType>& input_arr, array<ValueType>& result)
{
    GKO_ASSERT(result.get_size() == 1);
    auto exec = input_arr.get_executor();
    exec->run(array_kernels::make_reduce_add_array(input_arr, result));
}


template <typename ValueType>
ValueType reduce_add(const array<ValueType>& input_arr,
                     const ValueType init_value)
{
    auto exec = input_arr.get_executor();
    auto value = array<ValueType>(exec, 1);
    value.fill(ValueType{0});
    exec->run(array_kernels::make_reduce_add_array(input_arr, value));
    return init_value + get_element(value, 0);
}


#define GKO_DECLARE_ARRAY_FILL(_type) void array<_type>::fill(const _type value)

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_FILL);


#define GKO_DECLARE_ARRAY_REDUCE_ADD(_type) \
    void reduce_add(const array<_type>& arr, array<_type>& value)

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_REDUCE_ADD);


#define GKO_DECLARE_ARRAY_REDUCE_ADD2(_type) \
    _type reduce_add(const array<_type>& arr, const _type val)

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_REDUCE_ADD2);


}  // namespace gko
