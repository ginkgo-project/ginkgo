/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_ACCESSOR_CUDA_HELPER_HPP_
#define GKO_ACCESSOR_CUDA_HELPER_HPP_


#include <type_traits>


#include <thrust/complex.h>


#include "block_col_major.hpp"
#include "reduced_row_major.hpp"
#include "row_major.hpp"
#include "scaled_reduced_row_major.hpp"
#include "utils.hpp"


namespace gko {
namespace acc {
namespace detail {


template <typename T>
struct cuda_type {
    using type = T;
};

template <>
struct cuda_type<gko::half> {
    using type = __half;
};

// Unpack cv and reference / pointer qualifiers
template <typename T>
struct cuda_type<const T> {
    using type = const typename cuda_type<T>::type;
};

template <typename T>
struct cuda_type<volatile T> {
    using type = volatile typename cuda_type<T>::type;
};

template <typename T>
struct cuda_type<T*> {
    using type = typename cuda_type<T>::type*;
};

template <typename T>
struct cuda_type<T&> {
    using type = typename cuda_type<T>::type&;
};

template <typename T>
struct cuda_type<T&&> {
    using type = typename cuda_type<T>::type&&;
};


// Transform std::complex to thrust::complex
template <typename T>
struct cuda_type<std::complex<T>> {
    using type = thrust::complex<typename cuda_type<T>::type>;
};


}  // namespace detail


/**
 * This is an alias for CUDA's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using cuda_type_t = typename detail::cuda_type<T>::type;


/**
 * Reinterprets the passed in value as a CUDA type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA type
 */
template <typename T>
std::enable_if_t<std::is_pointer<T>::value || std::is_reference<T>::value,
                 cuda_type_t<T>>
as_cuda_type(T val)
{
    return reinterpret_cast<cuda_type_t<T>>(val);
}


/**
 * @copydoc as_cuda_type()
 */
template <typename T>
std::enable_if_t<!std::is_pointer<T>::value && !std::is_reference<T>::value,
                 cuda_type_t<T>>
as_cuda_type(T val)
{
    return *reinterpret_cast<cuda_type_t<T>*>(&val);
}


/**
 * Changes the types and reinterprets the passed in range pointers as a CUDA
 * types.
 *
 * @param r  the range which pointers need to be reinterpreted
 *
 * @return `r` with appropriate types and reinterpreted to CUDA pointers
 */
template <std::size_t dim, typename Type1, typename Type2>
GKO_ACC_INLINE auto as_cuda_range(
    const range<reduced_row_major<dim, Type1, Type2>>& r)
{
    return range<
        reduced_row_major<dim, cuda_type_t<Type1>, cuda_type_t<Type2>>>(
        r.get_accessor().get_size(),
        as_cuda_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_stride());
}

/**
 * @copydoc as_cuda_range()
 */
template <std::size_t dim, typename Type1, typename Type2, std::uint64_t mask>
GKO_ACC_INLINE auto as_cuda_range(
    const range<scaled_reduced_row_major<dim, Type1, Type2, mask>>& r)
{
    return range<scaled_reduced_row_major<dim, cuda_type_t<Type1>,
                                          cuda_type_t<Type2>, mask>>(
        r.get_accessor().get_size(),
        as_cuda_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_storage_stride(),
        as_cuda_type(r.get_accessor().get_scalar()),
        r.get_accessor().get_scalar_stride());
}

/**
 * @copydoc as_cuda_range()
 */
template <typename T, size_type dim>
GKO_ACC_INLINE auto as_cuda_range(const range<block_col_major<T, dim>>& r)
{
    return range<block_col_major<cuda_type_t<T>, dim>>(
        r.get_accessor().lengths, as_cuda_type(r.get_accessor().data),
        r.get_accessor().stride);
}

/**
 * @copydoc as_cuda_range()
 */
template <typename T, size_type dim>
GKO_ACC_INLINE auto as_cuda_range(const range<row_major<T, dim>>& r)
{
    return range<block_col_major<cuda_type_t<T>, dim>>(
        r.get_accessor().lengths, as_cuda_type(r.get_accessor().data),
        r.get_accessor().stride);
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_CUDA_HELPER_HPP_
