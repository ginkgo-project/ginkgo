// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_UTILS_HPP_
#define GKO_CORE_BASE_UTILS_HPP_


#include <memory>
#include <tuple>
#include <type_traits>

#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


template <typename ValueType, typename IndexType>
GKO_ATTRIBUTES GKO_INLINE ValueType checked_load(const ValueType* p,
                                                 IndexType i, IndexType size,
                                                 ValueType sentinel)
{
    return i < size ? p[i] : sentinel;
}


}  // namespace kernels


namespace detail {


template <typename Dest>
struct conversion_sort_helper {};

template <typename ValueType, typename IndexType>
struct conversion_sort_helper<matrix::Csr<ValueType, IndexType>> {
    using mtx_type = matrix::Csr<ValueType, IndexType>;
    template <typename Source>
    static std::unique_ptr<mtx_type> get_sorted_conversion(
        std::shared_ptr<const Executor>& exec, Source* source)
    {
        auto editable_mtx = mtx_type::create(exec);
        as<ConvertibleTo<mtx_type>>(source)->convert_to(editable_mtx);
        editable_mtx->sort_by_column_index();
        return editable_mtx;
    }
};


template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor>& exec, Source* obj, bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj);
        return {sorted_mtx.release(), std::default_delete<Dest>()};
    }
}

template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting_impl(
    std::shared_ptr<const Executor>& exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    if (skip_sorting) {
        return copy_and_convert_to<Dest>(exec, obj);
    } else {
        using decay_dest = std::decay_t<Dest>;
        auto sorted_mtx =
            detail::conversion_sort_helper<decay_dest>::get_sorted_conversion(
                exec, obj.get());
        return {std::move(sorted_mtx)};
    }
}


}  // namespace detail


/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned with an empty
 * deleter.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, Source* obj, bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::unique_ptr<const Dest, std::function<void(const Dest*)>>
convert_to_with_sorting(std::shared_ptr<const Executor> exec, const Source* obj,
                        bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * Source *, bool)
 *
 * @note This version has a unique_ptr as the source instead of a plain pointer
 */
template <typename Dest, typename Source>
std::unique_ptr<Dest, std::function<void(Dest*)>> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, const std::unique_ptr<Source>& obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj.get(),
                                                      skip_sorting);
}

/**
 * @internal
 *
 * Helper function that converts the given matrix to the Dest format with
 * additional sorting if requested.
 *
 * If the given matrix was already sorted, is on the same executor and with a
 * dynamic type of `Dest`, the same pointer is returned.
 * In all other cases, a new matrix is created, which stores the converted
 * matrix.
 *
 * @tparam Dest  the type to which the object should be converted
 * @tparam Source  the type of the source object
 *
 * @param exec  the executor where the result should be placed
 * @param obj  the source object that should be converted
 * @param skip_sorting  indicator if the resulting matrix should be sorted or
 *                      not
 */
template <typename Dest, typename Source>
std::shared_ptr<Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<Dest>(exec, obj, skip_sorting);
}

/**
 * @copydoc convert_to_with_sorting(std::shared_ptr<const Executor>,
 * std::shared_ptr<Source>, bool)
 *
 * @note This version adds the const qualifier for the result since the input is
 *       also const
 */
template <typename Dest, typename Source>
std::shared_ptr<const Dest> convert_to_with_sorting(
    std::shared_ptr<const Executor> exec, std::shared_ptr<const Source> obj,
    bool skip_sorting)
{
    return detail::convert_to_with_sorting_impl<const Dest>(exec, obj,
                                                            skip_sorting);
}

/**
 * Converts the given arguments into an array of entries of the requested
 * template type.
 *
 * @tparam T  The requested type of entries in the output array.
 *
 * @param args  Entities to be filled into an array after casting to type T.
 */
template <typename T, typename... Args>
constexpr std::array<T, sizeof...(Args)> to_std_array(Args&&... args)
{
    return {static_cast<T>(args)...};
}


/**
 * Static (compile-time) for loop.
 * This can be used to write a for-like loop where the loop variable is
 * constexpr in the body.
 */
template <auto Start, auto End, auto Inc, class Functor1D>
constexpr void constexpr_for(Functor1D&& f)
{
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        constexpr_for<Start + Inc, End, Inc>(f);
    }
}

/**
 * Generates a tuple of types, where each type is the given template generator
 * instantiated with one of the given list of (scalar) types.
 *
 * @tparam Generator  A type that defines a `generate` template inside.
 * @param Types  A list (pack or tuple) of (scalar) types.
 */
template <typename Generator, typename... Types>
struct instantiation_tuple {
    using type = std::tuple<typename Generator::template generate<Types>...>;
};

// Specialization - handles std::tuple. See the instantiation_list above.
template <typename Generator, typename... Types>
struct instantiation_tuple<Generator, std::tuple<Types...>> {
    using type = std::tuple<typename Generator::template generate<Types>...>;
};

// Helper alias
template <typename T, typename... Types>
using instantiation_tuple_t = typename instantiation_tuple<T, Types...>::type;


/**
 * Generates a transformed tuple of types,
 * where each type is the given template generator
 * instantiated with one of the give list of (scalar) types.
 *
 * @tparam TypeTransformer  A type will be parameterized by the generated
 *                          template. An example is std::unique_ptr.
 * @tparam Generator  A type that defines a `generate` template inside.
 * @param Types  A list (pack or tuple) of (scalar) types.
 */
template <template <typename> class TypeTransformer, typename Generator,
          typename... Types>
struct transformed_instantiation_tuple {
    using type = std::tuple<
        TypeTransformer<typename Generator::template generate<Types>>...>;
};

// Specialization - handles std::tuple. See the instantiation_list above.
template <template <typename> class TypeTransformer, typename Generator,
          typename... Types>
struct transformed_instantiation_tuple<TypeTransformer, Generator,
                                       std::tuple<Types...>> {
    using type = std::tuple<
        TypeTransformer<typename Generator::template generate<Types>>...>;
};

// Helper alias
template <template <typename> class TypeTransformer, typename T,
          typename... Types>
using transformed_instantiation_tuple_t =
    typename transformed_instantiation_tuple<TypeTransformer, T,
                                             Types...>::type;


/**
 * Generates an instantiation of a given template.
 *
 * @tparam Template  A class template.
 */
template <template <typename> class Template>
struct generator {
    // Defines an instantiation.
    template <typename T>
    using generate = Template<T>;
};

// Wrapper for multi-parameter templates (first parameter varies)
template <template <typename, typename...> class Template,
          typename... FixedArgs>
struct generator_partial {
    // Defines an instantiation.
    template <typename T>
    using generate = Template<T, FixedArgs...>;
};

template <typename T>
using ptr_type = T*;

/**
 * Assigns a given value to the given index in a tuple.
 *
 * @tparam k  Position in the tuple to check against the runtime index.
 * @tparam ValueType  scalar type to assign to the tuple position.
 * @tparam Args  Types that make up the tuple.
 *
 * @param t  The tuple to be modified.
 * @param value  The value to be assigned.
 * @param idx  The runtime position of the tuple that should be assigned to.
 */
template <int k, typename ValueType, typename... Args>
void assign_value_to_tuple(std::tuple<Args...>& t, const ValueType& value,
                           const int idx)
{
    constexpr int len = sizeof...(Args);
    if constexpr (k < 0 || k >= len) {
        return;
    } else if constexpr (k == len - 1) {
        if (k == idx) {
            std::get<k>(t) = value;
        }
        return;
    } else {
        if (k == idx) {
            std::get<k>(t) = value;
        } else {
            assign_value_to_tuple<k + 1>(t, value, idx);
        }
    }
}

/**
 * Assigns a given value to the given location of the given index
 * in a tuple of arrays.
 */
template <int k, typename ValueType, typename... Args>
inline void assign_value_to_array_tuple(const std::tuple<Args...>& t,
                                        const ValueType& value, const int t_idx,
                                        const int loc)
{
    constexpr int len = sizeof...(Args);
    if constexpr (k < 0 || k >= len) {
        return;
    } else if constexpr (k == len - 1) {
        if (k == t_idx) {
            std::get<k>(t)[loc] = value;
        }
        return;
    } else {
        if (k == t_idx) {
            std::get<k>(t)[loc] = value;
        } else {
            assign_value_to_array_tuple<k + 1>(t, value, t_idx, loc);
        }
    }
}


}  // namespace gko


#endif  // GKO_CORE_BASE_UTILS_HPP_
