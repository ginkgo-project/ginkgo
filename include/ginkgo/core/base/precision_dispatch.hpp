// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_
#define GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_conversion.hpp>
#include <ginkgo/core/distributed/vector.hpp>
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
template <typename ValueType, typename Ptr>
detail::temporary_conversion<std::conditional_t<
    std::is_const<detail::pointee<Ptr>>::value, const matrix::Dense<ValueType>,
    matrix::Dense<ValueType>>>
make_temporary_conversion(Ptr&& matrix)
{
    using Pointee = detail::pointee<Ptr>;
    using Dense = matrix::Dense<ValueType>;
    using NextDense = matrix::Dense<next_precision<ValueType>>;
    using MaybeConstDense =
        std::conditional_t<std::is_const<Pointee>::value, const Dense, Dense>;
    auto result = detail::temporary_conversion<
        MaybeConstDense>::template create<NextDense>(matrix);
    if (!result) {
        GKO_NOT_SUPPORTED(*matrix);
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
void precision_dispatch(Function fn, Args*... linops)
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
void precision_dispatch_real_complex(Function fn, const LinOp* in, LinOp* out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>>*>(in));
    if (complex_to_real) {
        auto dense_in = make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out = make_temporary_conversion<to_complex<ValueType>>(out);
        using Dense = matrix::Dense<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dynamic_cast<const Dense*>(dense_in->create_real_view().get()),
           dynamic_cast<Dense*>(dense_out->create_real_view().get()));
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
void precision_dispatch_real_complex(Function fn, const LinOp* alpha,
                                     const LinOp* in, LinOp* out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>>*>(in));
    if (complex_to_real) {
        auto dense_in = make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out = make_temporary_conversion<to_complex<ValueType>>(out);
        auto dense_alpha = make_temporary_conversion<ValueType>(alpha);
        using Dense = matrix::Dense<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dense_alpha.get(),
           dynamic_cast<const Dense*>(dense_in->create_real_view().get()),
           dynamic_cast<Dense*>(dense_out->create_real_view().get()));
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
void precision_dispatch_real_complex(Function fn, const LinOp* alpha,
                                     const LinOp* in, const LinOp* beta,
                                     LinOp* out)
{
    // do we need to convert complex Dense to real Dense?
    // all real dense vectors are intra-convertible, thus by casting to
    // ConvertibleTo<matrix::Dense<>>, we can check whether a LinOp is a real
    // dense matrix:
    auto complex_to_real =
        !(is_complex<ValueType>() ||
          dynamic_cast<const ConvertibleTo<matrix::Dense<>>*>(in));
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
           dynamic_cast<const Dense*>(dense_in->create_real_view().get()),
           dense_beta.get(),
           dynamic_cast<Dense*>(dense_out->create_real_view().get()));
    } else {
        precision_dispatch<ValueType>(fn, alpha, in, beta, out);
    }
}


/**
 * Calls the given function with each given argument LinOp
 * converted into matrix::Dense<ValueType> as parameters.
 *
 * If GINKGO_MIXED_PRECISION is defined, this means that the function will be
 * called with its dynamic type as a static type, so the (templated/generic)
 * function will be instantiated with all pairs of Dense<ValueType> and
 * Dense<next_precision<ValueType>> parameter types, and the appropriate
 * overload will be called based on the dynamic type of the parameter.
 *
 * If GINKGO_MIXED_PRECISION is not defined, it will behave exactly like
 * precision_dispatch.
 *
 * @param fn  the given function. It will be called with one const and one
 *            non-const matrix::Dense<...> parameter based on the dynamic type
 *            of the inputs (GINKGO_MIXED_PRECISION) or of type
 *            matrix::Dense<ValueType> (no GINKGO_MIXED_PRECISION).
 * @param in  The first parameter to be cast (GINKGO_MIXED_PRECISION) or
 *            converted (no GINKGO_MIXED_PRECISION) and used to call `fn`.
 * @param out  The second parameter to be cast (GINKGO_MIXED_PRECISION) or
 *             converted (no GINKGO_MIXED_PRECISION) and used to call `fn`.
 *
 * @tparam ValueType  the value type to use for the parameters of `fn` (no
 *                    GINKGO_MIXED_PRECISION). With GINKGO_MIXED_PRECISION
 *                    enabled, it only matters whether this type is complex or
 *                    real.
 * @tparam Function  the function pointer, lambda or other functor type to call
 *                   with the converted arguments.
 */
template <typename ValueType, typename Function>
void mixed_precision_dispatch(Function fn, const LinOp* in, LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    using fst_type = matrix::Dense<ValueType>;
    using snd_type = matrix::Dense<next_precision<ValueType>>;
    if (auto dense_in = dynamic_cast<const fst_type*>(in)) {
        if (auto dense_out = dynamic_cast<fst_type*>(out)) {
            fn(dense_in, dense_out);
        } else if (auto dense_out = dynamic_cast<snd_type*>(out)) {
            fn(dense_in, dense_out);
        } else {
            GKO_NOT_SUPPORTED(out);
        }
    } else if (auto dense_in = dynamic_cast<const snd_type*>(in)) {
        if (auto dense_out = dynamic_cast<fst_type*>(out)) {
            fn(dense_in, dense_out);
        } else if (auto dense_out = dynamic_cast<snd_type*>(out)) {
            fn(dense_in, dense_out);
        } else {
            GKO_NOT_SUPPORTED(out);
        }
    } else {
        GKO_NOT_SUPPORTED(in);
    }
#else
    precision_dispatch<ValueType>(fn, in, out);
#endif
}


/**
 * Calls the given function with the given LinOps cast to their dynamic type
 * matrix::Dense<ValueType>* as parameters.
 * If ValueType is real and both `in` and `out` are complex, uses
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see mixed_precision_dispatch()
 */
template <typename ValueType, typename Function,
          std::enable_if_t<is_complex<ValueType>()>* = nullptr>
void mixed_precision_dispatch_real_complex(Function fn, const LinOp* in,
                                           LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    mixed_precision_dispatch<ValueType>(fn, in, out);
#else
    precision_dispatch<ValueType>(fn, in, out);
#endif
}


template <typename ValueType, typename Function,
          std::enable_if_t<!is_complex<ValueType>()>* = nullptr>
void mixed_precision_dispatch_real_complex(Function fn, const LinOp* in,
                                           LinOp* out)
{
#ifdef GINKGO_MIXED_PRECISION
    if (!dynamic_cast<const ConvertibleTo<matrix::Dense<>>*>(in)) {
        mixed_precision_dispatch<to_complex<ValueType>>(
            [&fn](auto dense_in, auto dense_out) {
                fn(dense_in->create_real_view().get(),
                   dense_out->create_real_view().get());
            },
            in, out);
    } else {
        mixed_precision_dispatch<ValueType>(fn, in, out);
    }
#else
    precision_dispatch_real_complex<ValueType>(fn, in, out);
#endif
}


namespace experimental {


#if GINKGO_BUILD_MPI


namespace distributed {


/**
 * Convert the given LinOp from experimental::distributed::Vector<...> to
 * experimental::distributed::Vector<ValueType>. The conversion tries to convert
 * the input LinOp to all Dense types with value type recursively reachable by
 * next_precision<...> starting from the ValueType template parameter. This
 * means that all real-to-real and complex-to-complex conversions for default
 * precisions are being considered. If the input matrix is non-const, the
 * contents of the modified converted object will be converted back to the input
 * matrix when the returned object is destroyed. This may lead to a loss of
 * precision!
 *
 * @param matrix  the input matrix which is supposed to be converted. It is
 *                wrapped unchanged if it is already of type
 *                experimental::distributed::Vector<ValueType>, otherwise it
 * will be converted to this type if possible.
 *
 * @returns  a detail::temporary_conversion pointing to the (potentially
 *           converted) object.
 *
 * @throws NotSupported  if the input matrix cannot be converted to
 *                       experimental::distributed::Vector<ValueType>
 *
 * @tparam ValueType  the value type into whose associated
 * experimental::distributed::Vector type to convert the input LinOp.
 */
template <typename ValueType>
detail::temporary_conversion<experimental::distributed::Vector<ValueType>>
make_temporary_conversion(LinOp* matrix)
{
    auto result = detail::temporary_conversion<
        experimental::distributed::Vector<ValueType>>::
        template create<
            experimental::distributed::Vector<next_precision<ValueType>>>(
            matrix);
    if (!result) {
        GKO_NOT_SUPPORTED(matrix);
    }
    return result;
}


/**
 * @copydoc make_temporary_conversion
 */
template <typename ValueType>
detail::temporary_conversion<const experimental::distributed::Vector<ValueType>>
make_temporary_conversion(const LinOp* matrix)
{
    auto result = detail::temporary_conversion<
        const experimental::distributed::Vector<ValueType>>::
        template create<
            experimental::distributed::Vector<next_precision<ValueType>>>(
            matrix);
    if (!result) {
        GKO_NOT_SUPPORTED(matrix);
    }
    return result;
}


/**
 * Calls the given function with each given argument LinOp temporarily
 * converted into experimental::distributed::Vector<ValueType> as parameters.
 *
 * @param fn  the given function. It will be passed one (potentially const)
 *            experimental::distributed::Vector<ValueType>* parameter per
 * parameter in the parameter pack `linops`.
 * @param linops  the given arguments to be converted and passed on to fn.
 *
 * @tparam ValueType  the value type to use for the parameters of `fn`.
 * @tparam Function  the function pointer, lambda or other functor type to call
 *                   with the converted arguments.
 * @tparam Args  the argument type list.
 */
template <typename ValueType, typename Function, typename... Args>
void precision_dispatch(Function fn, Args*... linops)
{
    fn(distributed::make_temporary_conversion<ValueType>(linops).get()...);
}


/**
 * Calls the given function with the given LinOps temporarily converted to
 * experimental::distributed::Vector<ValueType>* as parameters.
 * If ValueType is real and both input vectors are complex, uses
 * experimental::distributed::Vector::get_real_view() to convert them into real
 * matrices after precision conversion.
 *
 * @see precision_dispatch()
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp* in, LinOp* out)
{
    auto complex_to_real = !(
        is_complex<ValueType>() ||
        dynamic_cast<const ConvertibleTo<experimental::distributed::Vector<>>*>(
            in));
    if (complex_to_real) {
        auto dense_in =
            distributed::make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out =
            distributed::make_temporary_conversion<to_complex<ValueType>>(out);
        using Vector = experimental::distributed::Vector<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dynamic_cast<const Vector*>(dense_in->create_real_view().get()),
           dynamic_cast<Vector*>(dense_out->create_real_view().get()));
    } else {
        distributed::precision_dispatch<ValueType>(fn, in, out);
    }
}


/**
 * @copydoc precision_dispatch_real_complex(Function, const LinOp*, LinOp*)
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp* alpha,
                                     const LinOp* in, LinOp* out)
{
    auto complex_to_real = !(
        is_complex<ValueType>() ||
        dynamic_cast<const ConvertibleTo<experimental::distributed::Vector<>>*>(
            in));
    if (complex_to_real) {
        auto dense_in =
            distributed::make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out =
            distributed::make_temporary_conversion<to_complex<ValueType>>(out);
        auto dense_alpha = gko::make_temporary_conversion<ValueType>(alpha);
        using Vector = experimental::distributed::Vector<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dense_alpha.get(),
           dynamic_cast<const Vector*>(dense_in->create_real_view().get()),
           dynamic_cast<Vector*>(dense_out->create_real_view().get()));
    } else {
        fn(gko::make_temporary_conversion<ValueType>(alpha).get(),
           distributed::make_temporary_conversion<ValueType>(in).get(),
           distributed::make_temporary_conversion<ValueType>(out).get());
    }
}


/**
 * @copydoc precision_dispatch_real_complex(Function, const LinOp*, LinOp*)
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex(Function fn, const LinOp* alpha,
                                     const LinOp* in, const LinOp* beta,
                                     LinOp* out)
{
    auto complex_to_real = !(
        is_complex<ValueType>() ||
        dynamic_cast<const ConvertibleTo<experimental::distributed::Vector<>>*>(
            in));
    if (complex_to_real) {
        auto dense_in =
            distributed::make_temporary_conversion<to_complex<ValueType>>(in);
        auto dense_out =
            distributed::make_temporary_conversion<to_complex<ValueType>>(out);
        auto dense_alpha = gko::make_temporary_conversion<ValueType>(alpha);
        auto dense_beta = gko::make_temporary_conversion<ValueType>(beta);
        using Vector = experimental::distributed::Vector<ValueType>;
        // These dynamic_casts are only needed to make the code compile
        // If ValueType is complex, this branch will never be taken
        // If ValueType is real, the cast is a no-op
        fn(dense_alpha.get(),
           dynamic_cast<const Vector*>(dense_in->create_real_view().get()),
           dense_beta.get(),
           dynamic_cast<Vector*>(dense_out->create_real_view().get()));
    } else {
        fn(gko::make_temporary_conversion<ValueType>(alpha).get(),
           distributed::make_temporary_conversion<ValueType>(in).get(),
           gko::make_temporary_conversion<ValueType>(beta).get(),
           distributed::make_temporary_conversion<ValueType>(out).get());
    }
}


}  // namespace distributed


/**
 * Calls the given function with the given LinOps temporarily converted to
 * either experimental::distributed::Vector<ValueType>* or
 * matrix::Dense<ValueType> as parameters. The choice depends on the runtime
 * type of `in` and `out` is assumed to fall into the same category. If
 * ValueType is real and both input vectors are complex, uses
 * experimental::distributed::Vector::get_real_view(), or
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see precision_dispatch()
 * @see distributed::precision_dispatch()
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex_distributed(Function fn, const LinOp* in,
                                                 LinOp* out)
{
    if (dynamic_cast<const experimental::distributed::DistributedBase*>(in)) {
        experimental::distributed::precision_dispatch_real_complex<ValueType>(
            fn, in, out);
    } else {
        gko::precision_dispatch_real_complex<ValueType>(fn, in, out);
    }
}


/**
 * @copydoc precision_dispatch_real_complex_distributed(Function, const LinOp*,
 * LinOp*)
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex_distributed(Function fn,
                                                 const LinOp* alpha,
                                                 const LinOp* in, LinOp* out)
{
    if (dynamic_cast<const experimental::distributed::DistributedBase*>(in)) {
        experimental::distributed::precision_dispatch_real_complex<ValueType>(
            fn, alpha, in, out);
    } else {
        gko::precision_dispatch_real_complex<ValueType>(fn, alpha, in, out);
    }
}


/**
 * @copydoc precision_dispatch_real_complex_distributed(Function, const LinOp*,
 * LinOp*)
 */
template <typename ValueType, typename Function>
void precision_dispatch_real_complex_distributed(Function fn,
                                                 const LinOp* alpha,
                                                 const LinOp* in,
                                                 const LinOp* beta, LinOp* out)
{
    if (dynamic_cast<const experimental::distributed::DistributedBase*>(in)) {
        experimental::distributed::precision_dispatch_real_complex<ValueType>(
            fn, alpha, in, beta, out);
    } else {
        gko::precision_dispatch_real_complex<ValueType>(fn, alpha, in, beta,
                                                        out);
    }
}


#else


/**
 * Calls the given function with the given LinOps temporarily converted to
 * matrix::Dense<ValueType> as parameters.
 * If ValueType is real and both input vectors are complex, uses
 * experimental::distributed::Vector::get_real_view(), or
 * matrix::Dense::get_real_view() to convert them into real matrices after
 * precision conversion.
 *
 * @see precision_dispatch()
 */
template <typename ValueType, typename Function, typename... Args>
void precision_dispatch_real_complex_distributed(Function fn, Args*... args)
{
    precision_dispatch_real_complex<ValueType>(fn, args...);
}


#endif


}  // namespace experimental
}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_PRECISION_DISPATCH_HPP_
