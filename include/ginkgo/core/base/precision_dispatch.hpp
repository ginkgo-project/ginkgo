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

#ifndef GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_
#define GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {


/**
 * Convert the given LinOp from matrix::Dense<...> to matrix::Dense<ValueType>.
 * The conversion tries to convert the input LinOp to all Dense types with value
 * type recursively reachable by next_precision<...> starting from the ValueType
 * template parameter. This means that all real-to-real and complex-to-complex
 * conversions for default precisions are being considered.
 * If the input matrix is non-const, the contents of the modified converted
 * object will be converted back to the input matrix when the returned object is
 * destroyed. This may lead to a loss of precision!
 *
 * @param matrix  the input matrix which is supposed to be converted. It is
 *                wrapped unchanged if it is already of type
 *                matrix::Dense<ValueType>, otherwise it will be converted to
 *                this type if possible.
 *
 * @returns  a detail::temporary_conversion pointing to the (potentially
 *           converted) object.
 *
 * @throws NotSupported  if the input matrix cannot be converted to
 *                       matrix::Dense<ValueType>
 *
 * @tparam ValueType  the value type into whose associated matrix::Dense type to
 *                    convert the input LinOp.
 */
template <typename ValueType>
detail::temporary_conversion<matrix::Dense<ValueType>>
make_temporary_conversion(LinOp *matrix)
{
    auto result =
        detail::temporary_conversion<matrix::Dense<ValueType>>::template create<
            matrix::Dense<next_precision<ValueType>>>(matrix);
    if (!result) {
        GKO_NOT_SUPPORTED(matrix);
    }
    return result;
}


/** @copydoc make_temporary_conversion(LinOp*) */
template <typename ValueType>
detail::temporary_conversion<const matrix::Dense<ValueType>>
make_temporary_conversion(const LinOp *matrix)
{
    auto result = detail::temporary_conversion<const matrix::Dense<ValueType>>::
        template create<matrix::Dense<next_precision<ValueType>>>(matrix);
    if (!result) {
        GKO_NOT_SUPPORTED(matrix);
    }
    return result;
}


/**
 * Calls the given function with each given argument LinOp temporarily
 * converted into matrix::Dense<ValueType> as parameters.
 *
 * @param fn  the given function. It will be passed one (potentially const)
 *            matrix::Dense<ValueType>* parameter per parameter in the parameter
 *            pack `linops`.
 * @param linops  the given arguments to be converted and passed on to fn.
 *
 * @tparam ValueType  the value type to use for the parameters of `fn`.
 * @tparam Function  the function pointer, lambda or other functor type to call
 *                   with the converted arguments.
 * @tparam Args  the argument type list.
 * */
template <typename ValueType, typename Function, typename... Args>
void precision_dispatch(Function fn, Args *... linops)
{
    fn(make_temporary_conversion<ValueType>(linops).get()...);
}


/**
 * Calls the given function with the given LinOps temporarily converted to
 * matrix::Dense<ValueType>* as parameters.
 * If ValueType is real and both input vectors are complex, uses
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see precision_dispatch()
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp *in, LinOp *out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>> *>(in));
    if (complex_to_real) {
        auto dense_in = make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out = make_temporary_conversion<to_complex<ValueType>>(out);
        using Dense = matrix::Dense<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dynamic_cast<const Dense *>(dense_in->create_real_view().get()),
           dynamic_cast<Dense *>(dense_out->create_real_view().get()));
    } else {
        precision_dispatch<ValueType>(fn, in, out);
    }
}


/**
 * Calls the given function with the given LinOps temporarily converted to
 * matrix::Dense<ValueType>* as parameters.
 * If ValueType is real and both `in` and `out` are complex, uses
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see precision_dispatch()
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp *alpha,
                                     const LinOp *in, LinOp *out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>> *>(in));
    if (complex_to_real) {
        auto dense_in = make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out = make_temporary_conversion<to_complex<ValueType>>(out);
        auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
        using Dense = matrix::Dense<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dense_alpha.get(),
           dynamic_cast<const Dense *>(dense_in->create_real_view().get()),
           dynamic_cast<Dense *>(dense_out->create_real_view().get()));
    } else {
        precision_dispatch<ValueType>(fn, alpha, in, out);
    }
}


/**
 * Calls the given function with the given LinOps temporarily converted to
 * matrix::Dense<ValueType>* as parameters.
 * If ValueType is real and both `in` and `out` are complex, uses
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see precision_dispatch()
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp *alpha,
                                     const LinOp *in, const LinOp *beta,
                                     LinOp *out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>> *>(in));
    if (complex_to_real) {
        auto dense_in = make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out = make_temporary_conversion<to_complex<ValueType>>(out);
        auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
        auto dense_beta = make_temporary_conversion<ValueType>(beta);
        using Dense = matrix::Dense<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dense_alpha.get(),
           dynamic_cast<const Dense *>(dense_in->create_real_view().get()),
           dense_beta.get(),
           dynamic_cast<Dense *>(dense_out->create_real_view().get()));
    } else {
        precision_dispatch<ValueType>(fn, alpha, in, beta, out);
    }
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_
