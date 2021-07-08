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

#include <ginkgo/core/base/array.hpp>


#include <ginkgo/core/base/math.hpp>


#include "core/components/accumulate_array.hpp"
#include "core/components/fill_array.hpp"
#include "core/components/precision_conversion.hpp"
#include "core/components/reduce_array.hpp"


namespace gko {
namespace conversion {
namespace {


GKO_REGISTER_OPERATION(convert, components::convert_precision);


}  // anonymous namespace
}  // namespace conversion


namespace array {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);
GKO_REGISTER_OPERATION(accumulate_array, components::accumulate_array);
GKO_REGISTER_OPERATION(reduce_array, components::reduce_array);


}  // anonymous namespace
}  // namespace array


namespace detail {


template <typename SourceType, typename TargetType>
void convert_data(std::shared_ptr<const Executor> exec, size_type size,
                  const SourceType *src, TargetType *dst)
{
    exec->run(conversion::make_convert(size, src, dst));
}


#define GKO_DECLARE_ARRAY_CONVERSION(From, To)                              \
    void convert_data<From, To>(std::shared_ptr<const Executor>, size_type, \
                                const From *, To *)

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION(GKO_DECLARE_ARRAY_CONVERSION);


}  // namespace detail


template <typename ValueType>
void Array<ValueType>::fill(const ValueType value)
{
    this->get_executor()->run(
        array::make_fill_array(this->get_data(), this->get_num_elems(), value));
}


template <typename ValueType>
void Array<ValueType>::reduce(ValueType *value) const
{
    this->get_executor()->run(array::make_reduce_array(
        this->get_const_data(), this->get_num_elems(), value));
}


#define GKO_DECLARE_ARRAY_FILL(_type) void Array<_type>::fill(const _type value)

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_FILL);


template <typename ValueType>
ValueType Array<ValueType>::accumulate(const ValueType value) const
{
    auto exec = this->get_executor();
    Array<ValueType> accumulate(exec, {0});
    exec->run(array::make_accumulate_array(accumulate.get_data(),
                                           this->get_const_data(),
                                           this->get_num_elems(), value));
    accumulate.set_executor(exec->get_master());
    return accumulate.get_const_data()[0];
}


#define GKO_DECLARE_ARRAY_ACCUMULATE(_type) \
    _type Array<_type>::accumulate(const _type value) const

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_ACCUMULATE);


#define GKO_DECLARE_ARRAY_REDUCE(_type) \
    void Array<_type>::reduce(_type *value) const

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_ARRAY_REDUCE);


}  // namespace gko
