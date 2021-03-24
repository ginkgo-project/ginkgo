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

#include <tuple>


#include <ginkgo/core/base/range.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace syn {
namespace detail {


template <typename KernelFunctor, typename... Args>
struct kernel_2d_impl {
    KernelFunctor fn;
    dim<2> size;
    std::tuple<Args...> args;
};


template <typename T>
struct pointwise_wrapper {
    T *obj;
};


template <typename T>
struct rowwise_wrapper {
    T *obj;
};


template <typename T>
struct colwise_wrapper {
    T *obj;
};


template <typename T>
struct scalar_wrapper {
    T *obj;
};


struct row_wrapper {};


struct col_wrapper {};


template <typename T>
struct tuple_wrap_helper {
    static std::tuple<T> wrap(T obj) { return std::make_tuple(obj); }
};

template <typename... Ts>
struct tuple_wrap_helper<std::tuple<Ts...>> {
    static std::tuple<Ts...> wrap(std::tuple<Ts...> obj) { return obj; }
};


}  // namespace detail


inline std::tuple<detail::row_wrapper> row() { return {}; }


inline std::tuple<detail::col_wrapper> col() { return {}; }


inline std::tuple<detail::row_wrapper, detail::col_wrapper> row_col()
{
    return {};
}


template <typename... Ts>
std::tuple<detail::pointwise_wrapper<Ts>...> pointwise(Ts *... args)
{
    return std::make_tuple(detail::pointwise_wrapper<Ts>{args}...);
}


template <typename... Ts>
std::tuple<detail::rowwise_wrapper<Ts>...> rowwise(Ts *... args)
{
    return std::make_tuple(detail::rowwise_wrapper<Ts>{args}...);
}


template <typename... Ts>
std::tuple<detail::colwise_wrapper<Ts>...> colwise(Ts *... args)
{
    return std::make_tuple(detail::colwise_wrapper<Ts>{args}...);
}


template <typename... Ts>
std::tuple<detail::scalar_wrapper<Ts>...> scalar(Ts *... args)
{
    return std::make_tuple(detail::scalar_wrapper<Ts>{args}...);
}


template <typename KernelFunctor, typename... Args>
detail::kernel_2d_impl<KernelFunctor, Args...> kernel_2d_unpacked(
    KernelFunctor fn, dim<2> size, std::tuple<Args...> args)
{
    return {fn, size, args};
}


template <typename KernelFunctor, typename... Args>
decltype(auto) kernel_2d(KernelFunctor fn, dim<2> size, Args... args)
{
    return kernel_2d_unpacked(
        fn, size,
        std::tuple_cat(detail::tuple_wrap_helper<Args>::wrap(args)...));
}


}  // namespace syn
}  // namespace gko