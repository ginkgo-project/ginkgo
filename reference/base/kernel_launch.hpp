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

#include <utility>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace syn {
namespace detail {


template <typename ValueType>
struct matrix_accessor {
    ValueType *data;
    size_type stride;

    ValueType &operator()(size_type row, size_type col)
    {
        return data[row * stride + col];
    }
};


template <typename Tuple, typename Functor, std::size_t... Is>
decltype(auto) map_tuple_impl(Tuple &&tuple, Functor fn,
                              std::index_sequence<Is...>)
{
    return std::make_tuple(fn(std::get<Is>(tuple))...);
}


template <typename Tuple, typename Functor, std::size_t... Is>
decltype(auto) invoke_tuple_impl(Tuple &&tuple, Functor fn,
                                 std::index_sequence<Is...>)
{
    return fn(std::get<Is>(tuple)...);
}


}  // namespace detail


template <typename... Ts, typename Functor>
decltype(auto) map_tuple(const std::tuple<Ts...> &tuple, Functor fn)
{
    return detail::map_tuple_impl(tuple, fn,
                                  std::index_sequence_for<Ts &...>{});
}


template <typename... Ts, typename Functor>
decltype(auto) invoke_tuple(const std::tuple<Ts...> &tuple, Functor fn)
{
    return detail::invoke_tuple_impl(tuple, fn,
                                     std::index_sequence_for<Ts...>{});
}


// 2D accessors for matrices
template <typename T>
T &map_reference_2d(detail::rowwise_wrapper<matrix::Dense<T>> &obj,
                    size_type row, size_type col)
{
    return obj.obj->at(row, 0);
}


template <typename T>
const T &map_reference_2d(detail::rowwise_wrapper<const matrix::Dense<T>> &obj,
                          size_type row, size_type col)
{
    return obj.obj->get_const_values()[row * obj.obj->get_stride()];
}


template <typename T>
T &map_reference_2d(detail::colwise_wrapper<matrix::Dense<T>> &obj,
                    size_type row, size_type col)
{
    return obj.obj->at(0, col);
}


template <typename T>
const T &map_reference_2d(detail::colwise_wrapper<const matrix::Dense<T>> &obj,
                          size_type row, size_type col)
{
    return obj.obj->get_const_values()[col];
}


template <typename T>
T &map_reference_2d(detail::pointwise_wrapper<matrix::Dense<T>> &obj,
                    size_type row, size_type col)
{
    return obj.obj->at(row, col);
}


template <typename T>
const T &map_reference_2d(
    detail::pointwise_wrapper<const matrix::Dense<T>> &obj, size_type row,
    size_type col)
{
    return obj.obj->get_const_values()[col + obj.obj->get_stride() * row];
}


template <typename T>
detail::matrix_accessor<T> map_reference_2d(matrix::Dense<T> *obj,
                                            size_type row, size_type col)
{
    return detail::matrix_accessor<T>{obj->get_values(), obj->get_stride()};
}


template <typename T>
detail::matrix_accessor<const T> map_reference_2d(const matrix::Dense<T> *obj,
                                                  size_type row, size_type col)
{
    return detail::matrix_accessor<const T>{obj->get_const_values(),
                                            obj->get_stride()};
}


// 2D accessors for arrays
template <typename T>
T &map_reference_2d(detail::rowwise_wrapper<Array<T>> &obj, size_type row,
                    size_type col)
{
    return obj.obj->get_data()[row];
}


template <typename T>
const T &map_reference_2d(detail::rowwise_wrapper<const Array<T>> &obj,
                          size_type row, size_type col)
{
    return obj.obj->get_const_data()[row];
}


template <typename T>
T &map_reference_2d(detail::colwise_wrapper<Array<T>> &obj, size_type row,
                    size_type col)
{
    return obj.obj->get_data()[col];
}


template <typename T>
const T &map_reference_2d(detail::colwise_wrapper<const Array<T>> &obj,
                          size_type row, size_type col)
{
    return obj.obj->get_const_data()[col];
}


template <typename T>
stopping_status *map_reference_2d(Array<T> *obj, size_type, size_type)
{
    return obj->get_data();
}


template <typename T>
const stopping_status *map_reference_2d(const Array<T> *obj, size_type,
                                        size_type)
{
    return obj->get_const_data();
}


// 2D accessors for raw pointers
template <typename T>
T &map_reference_2d(detail::rowwise_wrapper<T> &obj, size_type row,
                    size_type col)
{
    return obj[row];
}


template <typename T>
T &map_reference_2d(detail::colwise_wrapper<T> &obj, size_type row,
                    size_type col)
{
    return obj[col];
}


template <typename T>
T *map_reference_2d(T *obj, size_type, size_type)
{
    return obj;
}


size_type map_reference_2d(detail::row_wrapper, size_type row, size_type col)
{
    return row;
}


size_type map_reference_2d(detail::col_wrapper, size_type row, size_type col)
{
    return col;
}


template <typename KernelFunctor, typename... Args>
void run_reference_kernel(detail::kernel_2d_impl<KernelFunctor, Args...> kernel)
{
    for (size_type row = 0; row < kernel.size[0]; row++) {
        for (size_type col = 0; col < kernel.size[1]; col++) {
            invoke_tuple(kernel.args, [&](auto... args) {
                kernel.fn(map_reference_2d(args, row, col)...);
            });
        }
    }
}


}  // namespace syn


template <typename KernelDescription>
void ReferenceExecutor::run_kernel(KernelDescription kernel) const
{
    syn::run_reference_kernel(kernel);
}


}  // namespace gko