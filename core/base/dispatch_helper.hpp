// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DISPATCH_HELPER_HPP_
#define GKO_CORE_BASE_DISPATCH_HELPER_HPP_


#include <memory>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace detail {


template <typename T, typename MaybeConstU>
using with_same_constness_t = std::conditional_t<
    std::is_const<typename std::remove_reference_t<MaybeConstU>>::value,
    const T, T>;


/**
 *
 * @copydoc run<typename ReturnType, typename K, typename... Types, typename T,
 *              typename Func, typename... Args>(T*, Func&&, Args&&...)
 *
 * @note this is the end case
 */
template <typename ReturnType, typename T, typename Func, typename... Args>
ReturnType run_impl(T* obj, Func&&, Args&&...)
{
    GKO_NOT_SUPPORTED(obj);
}

/**
 * @copydoc run<typename ReturnType, typename K, typename... Types, typename T,
 *              typename Func, typename... Args>(T*, Func&&, Args&&...)
 *
 * @note This has additionally the return type encoded.
 */
template <typename ReturnType, typename K, typename... Types, typename T,
          typename Func, typename... Args>
ReturnType run_impl(T* obj, Func&& f, Args&&... args)
{
    if (auto dobj = dynamic_cast<with_same_constness_t<K, T>*>(obj)) {
        return f(dobj, std::forward<Args>(args)...);
    } else {
        return run_impl<ReturnType, Types...>(obj, std::forward<Func>(f),
                                              std::forward<Args>(args)...);
    }
}


/**
 * @copydoc run<template <typename> class Base, typename T, typename Func,
 *              typename... Args>(T, Func&&, Args&&... )
 *
 * @note This is the end case for the smart pointer cases
 */
template <typename ReturnType, typename T, typename Func, typename... Args>
ReturnType run_impl(T obj, Func, Args...)
{
    GKO_NOT_SUPPORTED(obj);
}


/**
 * @copydoc run<template <typename> class Base, typename T, typename Func,
 *              typename... Args>(T, Func&&, Args&&... )
 *
 * @note This handles the shared pointer case
 */
template <typename ReturnType, typename K, typename... Types, typename T,
          typename Func, typename... Args>
ReturnType run_impl(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    if (auto dobj =
            std::dynamic_pointer_cast<with_same_constness_t<K, T>>(obj)) {
        return f(dobj, args...);
    } else {
        return run_impl<ReturnType, Types...>(obj, std::forward<Func>(f),
                                              std::forward<Args>(args)...);
    }
}

/**
 * Helper struct to get the result type of a function.
 *
 * @tparam T  Blueprint type for the function. This determines the
 *            const-qualifier for K, as well as the pointer type
 *            (either T*, or shared_ptr<T>) for K.
 * @tparam K  The actual type to be used in the function.
 * @tparam Func  The function to get the result from.
 * @tparam Args  Additional arguments to the function.
 */
template <typename T, typename K, typename Func, typename... Args>
struct result_of;

template <typename T, typename K, typename Func, typename... Args>
struct result_of<T*, K, Func, Args...> {
#if __cplusplus < 201703L
    // result_of_t is deprecated in C++17
    using type =
        std::result_of_t<Func(detail::with_same_constness_t<K, T>*, Args...)>;
#else
    using type =
        std::invoke_result_t<Func, detail::with_same_constness_t<K, T>*,
                             Args...>;
#endif
};

template <typename T, typename K, typename Func, typename... Args>
struct result_of<std::shared_ptr<T>, K, Func, Args...> {
#if __cplusplus < 201703L
    // result_of_t is deprecated in C++17
    using type = std::result_of_t<Func(
        std::shared_ptr<detail::with_same_constness_t<K, T>>, Args...)>;
#else
    using type = std::invoke_result_t<
        Func, std::shared_ptr<detail::with_same_constness_t<K, T>>, Args...>;
#endif
};

template <typename T, typename K, typename Func, typename... Args>
using result_of_t = typename result_of<T, K, Func, Args...>::type;


}  // namespace detail


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam K  the current type tried in the conversion
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the type of input object
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to K
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note  This assumes that the return type of f is independent of the input
 *        types (K, Types...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
auto run(T* obj, Func&& f, Args&&... args)
{
    using ReturnType = detail::result_of_t<T*, K, Func, Args...>;
    return detail::run_impl<ReturnType, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam ...Types  the types that will be tried with Base, i.e. Base<Types>...
 * @tparam T  the type of input object waiting converted
 * @tparam Func  the function will run if the object can be converted to pointer
 *               of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object waiting converted
 * @param f  the function will run if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <template <class> class Base, typename... Types, typename T,
          typename Func, typename... Args>
auto run(T* obj, Func&& f, Args&&... args)
{
    return run<Base<Types>...>(obj, std::forward<Func>(f),
                               std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam K  the current type to try in the conversion
 * @tparam ...Types  the other types will be tried in the conversion if K fails
 * @tparam T  the element type of input object waiting converted
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to pointer of Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note   This assumes that the return type of f is independent of the input
 *         types (smart_ptr<K>, smart_ptr<Types>...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <typename K, typename... Types, typename T, typename Func,
          typename... Args>
auto run(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    using ReturnType =
        detail::result_of_t<std::shared_ptr<T>, K, Func, Args...>;
    return detail::run_impl<ReturnType, K, Types...>(
        obj, std::forward<Func>(f), std::forward<Args>(args)...);
}


/**
 * run uses template to go through the list and select the valid
 * template and run it.
 *
 * @tparam Base  the Base class with one template
 * @tparam ...Types  the types that will be tried with Base, i.e. Base<Types>...
 * @tparam T  the element type of input object waiting converted
 * @tparam Func  the function type that is invoked if the object can be
 *               converted to pointer of const Base<K>
 * @tparam ...Args  the additional arguments for the Func
 *
 * @param obj  the input object that should be converted
 * @param f  the function will get invoked if obj can be converted successfully
 * @param args  the additional arguments for the function
 *
 * @note   This assumes that the return type of f is independent of the input
 *         types (smart_ptr<Base<K>>, smart_ptr<Base<Types>>...)
 *
 * @return  the result of f invoked with obj cast to the first matching type
 */
template <template <typename> class Base, typename... Types, typename T,
          typename Func, typename... Args>
auto run(std::shared_ptr<T> obj, Func&& f, Args&&... args)
{
    return run<Base<Types>...>(obj, std::forward<Func>(f),
                               std::forward<Args>(args)...);
}

/**
 * Helper to dispatch vectors to the expected precision.
 * Also handles complex->real conversion if necessary.
 *
 * @tparam ValueType Value type to convert the inputs to
 * @tparam Fn Function type, has signature void(const MultiVector*,
 *                                              MultiVector*)
 *
 * @param fn Function to apply to the converted inputs
 * @param b Input vector
 * @param x Output vector
 */
template <typename ValueType, typename Fn>
void precision_dispatch(Fn&& fn, const MultiVector* b, MultiVector* x)
{
    auto p = type_to_precision<ValueType>;
    if constexpr (!is_complex<ValueType>()) {
        fn(b->create_real_view()->as_precision(p).get(),
           x->create_real_view()->as_precision(p).get());
    } else {
        fn(b->as_precision(p).get(), x->as_precision(p).get());
    }
}


/**
 * Specialization for precision_dispatch for operator apply.
 *
 * Note: the function needs to have the following signature:
 *       fn(device_view_type<ValueType>, device_view_type<ValueType>)
 */
template <typename ValueType, typename Fn>
void apply_precision_dispatch(Fn&& fn, const MultiVector* b, MultiVector* x)
{
    precision_dispatch<ValueType>(
        [&fn](auto b_, auto x_) {
            fn(b_->template get_const_local_device_view<ValueType>(),
               x_->template get_local_device_view<ValueType>());
        },
        b, x);
}


/**
 * Same as apply_dispatch(Fn, const MultiVector*, MultiVector*), except for the
 * additional alpha and beta scalars.
 *
 * Note: the function needs to have the following signature:
 *       fn(Dense<ValueType>*, device_view_type<ValueType>, Dense<ValueType>*,
 *          device_view_type<ValueType>)
 */
template <typename ValueType, typename Fn>
void apply_precision_dispatch(Fn&& fn, const MultiVector* alpha,
                              const MultiVector* b, const MultiVector* beta,
                              MultiVector* x)
{
    auto p = type_to_precision<ValueType>;
    auto dense_alpha = as<matrix::Dense<ValueType>>(alpha->as_precision(p));
    auto dense_beta = as<matrix::Dense<ValueType>>(beta->as_precision(p));
    precision_dispatch<ValueType>(
        [&fn, &dense_alpha, &dense_beta](auto b_, auto x_) {
            fn(dense_alpha.get(),
               b_->template get_const_local_device_view<ValueType>(),
               dense_beta.get(),
               x_->template get_local_device_view<ValueType>());
        },
        b, x);
}


/**
 * Helper function for mixed precision dispatch.
 * Falls back to apply_dispatch if GINKGO_MIXED_PRECISION is not defined.
 *
 * The input vectors will be _not_ be converted to the precision of the
 * operator. Instead, the underlying precision of each vector will be used.
 * Exception: If the operator is complex, the vectors will be converted to their
 *            corresponding real precision.
 *
 * @tparam ValueType Value type to ensure compatibility with
 * @tparam Fn Function type, has signature
 *            void(const MultiVector* b, MultiVector* x, ValueTypeIn,
 *                 ValueTypeOut)
 *
 * @param fn Function to apply to the inputs
 * @param b Input vector
 * @param x Output vector
 */
template <typename ValueType, typename Fn>
void mixed_precision_dispatch(Fn&& fn, const MultiVector* b, MultiVector* x)
{
#ifdef GINKGO_MIXED_PRECISION
    auto precision_b = precision_to_variant(b->get_precision());
    auto precision_x = precision_to_variant(x->get_precision());
    std::visit(
        [&fn, b, x](auto p_b, auto p_x) {
            using fst_value_type = std::decay_t<decltype(p_b)>;
            using snd_value_type = std::decay_t<decltype(p_x)>;
            if constexpr (is_complex<ValueType>() ==
                              is_complex<fst_value_type>() &&
                          is_complex<ValueType>() ==
                              is_complex<snd_value_type>()) {
                // Either all precisions are real or all precisions are complex
                fn(b, x, p_b, p_x);
            } else if constexpr (!is_complex<ValueType>() &&
                                 is_complex<fst_value_type>() &&
                                 is_complex<snd_value_type>()) {
                // ValueType is real and both other precisions are complex
                fn(b->create_real_view().get(), x->create_real_view().get(),
                   remove_complex<fst_value_type>(),
                   remove_complex<snd_value_type>());
            } else {
                // real ValueType and one real and one complex precision are not
                // supported
                GKO_NOT_IMPLEMENTED;
            }
        },
        precision_b, precision_x);
#else
    precision_dispatch<ValueType>(
        [&fn](auto b_, auto x_, auto...) {
            fn(b_, x_, ValueType(), ValueType());
        },
        b, x);
#endif
}

/**
 * Specialization for mixed_precision_dispatch for operator apply.
 *
 * Note: the function needs to have the following signature:
 *       fn(device_view_type<ValueTypeIn>, device_view_type<ValueTypeOut>,
 *          ValueTypeIn, ValueTypeOut)
 */
template <typename ValueType, typename Fn>
void apply_mixed_precision_dispatch(Fn&& fn, const MultiVector* b,
                                    MultiVector* x)
{
    mixed_precision_dispatch<ValueType>(
        [&fn](auto b_, auto x_, auto p_b, auto p_x) {
            using fst_value_type = std::decay_t<decltype(p_b)>;
            using snd_value_type = std::decay_t<decltype(p_x)>;
            fn(b_->template get_const_local_device_view<fst_value_type>(),
               x_->template get_local_device_view<snd_value_type>(), p_b, p_x);
        },
        b, x);
}


/**
 * Same as mixed_precision_apply_dispatch(Fn, const MultiVector*, MultiVector*),
 * except for the additional alpha and beta scalars.
 *
 * @note the function needs to have the following signature:
 *       fn(Dense<ValueType>, device_view_type<ValueTypeIn>,
 *          Dense<ValueTypeOut>, device_view_type<ValueTypeOut>,
 *          ValueTypeIn, ValueTypeOut)
 *
 * @param alpha input scalar converted to precision ValueType if necessary
 * @param beta input scalar converted to precision of x if necessary
 */
template <typename ValueType, typename Fn>
void apply_mixed_precision_dispatch(Fn&& fn, const MultiVector* alpha,
                                    const MultiVector* b,
                                    const MultiVector* beta, MultiVector* x)
{
    auto dense_alpha = as<matrix::Dense<ValueType>>(
        alpha->as_precision(type_to_precision<ValueType>));

    mixed_precision_dispatch<ValueType>(
        [&fn, &dense_alpha, beta](auto b_, auto x_, auto p_b, auto p_x) {
            using fst_value_type = std::decay_t<decltype(p_b)>;
            using snd_value_type = std::decay_t<decltype(p_x)>;
            auto dense_beta = as<matrix::Dense<snd_value_type>>(
                beta->as_precision(type_to_precision<snd_value_type>));
            fn(dense_alpha.get(),
               b_->template get_const_local_device_view<fst_value_type>(),
               dense_beta.get(),
               x_->template get_local_device_view<snd_value_type>(), p_b, p_x);
        },
        b, x);
}


}  // namespace gko

#endif  // GKO_CORE_BASE_DISPATCH_HELPER_HPP_
