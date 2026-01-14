// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_AMP_UTILS_H_
#define GKO_CORE_BASE_AMP_UTILS_H_

#include <tuple>


namespace gko {


/**
 * Static for loop
 */
template <auto Start, auto End, auto Inc, class Functor1D>
constexpr void constexpr_for(Functor1D&& f)
{
    if constexpr (Start < End) {
        f(std::integral_constant<decltype(Start), Start>());
        constexpr_for<Start + Inc, End, Inc>(f);
    }
}

// template <bool cond, int val>
// struct enable_if_v {};

// template <int val>
// struct enable_if_v<true, val> {
//     static constexpr int value = val;
// };

/**
 * Generates a tuple of types, where each type is the given template generator
 * instantiated with one of the give list of (scalar) types.
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
 */
template <int len, int k, typename ValueType, typename... Args>
void assign_value_to_tuple(std::tuple<Args...>& t, const ValueType& value,
                           const int idx)
{
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
            assign_value_to_tuple<len, k + 1>(t, value, idx);
        }
    }
}

/**
 * Assigns a given value to the given location of the given index
 * in a tuple of arrays.
 */
template <int len, int k, typename ValueType, typename... Args>
inline void assign_value_to_array_tuple(const std::tuple<Args...>& t,
                                        const ValueType& value, const int t_idx,
                                        const int loc)
{
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
            assign_value_to_array_tuple<len, k + 1>(t, value, t_idx, loc);
        }
    }
}


}  // namespace gko


#endif  // GKO_CORE_BASE_AMP_UTILS_H_
