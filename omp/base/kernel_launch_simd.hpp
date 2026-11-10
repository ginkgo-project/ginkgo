// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_BASE_KERNEL_LAUNCH_SIMD_HPP_
#define GKO_OMP_BASE_KERNEL_LAUNCH_SIMD_HPP_


#include <cstddef>
#include <type_traits>

#include <hwy/highway.h>

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {
namespace simd {


namespace hn = hwy::HWY_NAMESPACE;


/**
 * @internal
 * Index handed to a kernel body in place of the row index on the SIMD path.
 * No comparison or arithmetic operators: a body that inspects the row index is
 * unsafe to vectorize and must fail to compile instead.
 */
struct index {
    int64 base;
    std::size_t count;
};


template <typename X, typename = void>
struct is_expr_impl : std::false_type {};

template <typename X>
struct is_expr_impl<X, std::void_t<typename std::decay_t<X>::gko_simd_expr>>
    : std::true_type {};

/**
 * @internal
 * True for the expression-template nodes below. A nested typedef rather than a
 * base class, which would make them non-aggregates in C++17.
 */
template <typename X>
inline constexpr bool is_expr = is_expr_impl<X>::value;


template <typename A, typename B>
using tag_of_t =
    typename std::decay_t<std::conditional_t<is_expr<A>, A, B>>::tag;


/**
 * @internal
 * Access tags selecting the load/store form of the views below.
 * `partial_access` uses LoadN, which zero-fills and does not fault, so the
 * final partial iteration needs no scalar remainder loop.
 * Type tags rather than a `bool`: see the partial-ordering note on the helpers
 * at the bottom of this header.
 */
struct full_access {};
struct partial_access {};


/**
 * @internal
 * Read-only view of `count` contiguous lanes. A lazy alias into memory, not a
 * value -- see the warning on `ref` below.
 */
template <typename D, typename Access>
struct ref_leaf {
    using gko_simd_expr = void;
    using tag = D;

    const hn::TFromD<D>* ptr;
    std::size_t count;

    hn::Vec<D> eval(D d) const
    {
        if constexpr (std::is_same_v<Access, partial_access>) {
            return hn::LoadN(d, ptr, count);
        } else {
            return hn::LoadU(d, ptr);
        }
    }
};


/**
 * @internal
 * A loop-invariant scalar broadcast to all lanes. Stored by value, which makes
 * `auto tmp = ...` inside a kernel body safe rather than dangling.
 */
template <typename D, typename T>
struct scalar_leaf {
    using gko_simd_expr = void;
    using tag = D;

    T value;

    hn::Vec<D> eval(D d) const
    {
        return hn::Set(d, static_cast<hn::TFromD<D>>(value));
    }
};


#define GKO_SIMD_BINARY_NODE(_name, _op)              \
    template <typename D, typename Lhs, typename Rhs> \
    struct _name {                                    \
        using gko_simd_expr = void;                   \
        using tag = D;                                \
                                                      \
        Lhs lhs;                                      \
        Rhs rhs;                                      \
                                                      \
        hn::Vec<D> eval(D d) const                    \
        {                                             \
            return hn::_op(lhs.eval(d), rhs.eval(d)); \
        }                                             \
    }

GKO_SIMD_BINARY_NODE(mul_expr, Mul);
GKO_SIMD_BINARY_NODE(div_expr, Div);

#undef GKO_SIMD_BINARY_NODE


template <typename X>
struct is_mul_expr_impl : std::false_type {};

template <typename D, typename Lhs, typename Rhs>
struct is_mul_expr_impl<mul_expr<D, Lhs, Rhs>> : std::true_type {};

/** @internal True for a multiply node, the shape the fused ops look for. */
template <typename X>
inline constexpr bool is_mul_expr = is_mul_expr_impl<std::decay_t<X>>::value;


/*
 * add_expr, sub_expr and ref's compound assignments emit MulAdd/NegMulAdd when
 * a multiply feeds them, matching the FMA the scalar path contracts into.
 */
template <typename D, typename Lhs, typename Rhs>
struct add_expr {
    using gko_simd_expr = void;
    using tag = D;

    Lhs lhs;
    Rhs rhs;

    hn::Vec<D> eval(D d) const
    {
        if constexpr (is_mul_expr<Lhs>) {
            return hn::MulAdd(lhs.lhs.eval(d), lhs.rhs.eval(d), rhs.eval(d));
        } else if constexpr (is_mul_expr<Rhs>) {
            return hn::MulAdd(rhs.lhs.eval(d), rhs.rhs.eval(d), lhs.eval(d));
        } else {
            return hn::Add(lhs.eval(d), rhs.eval(d));
        }
    }
};


/** @internal Only a multiply on the right fuses; `a * b - c` stays separate. */
template <typename D, typename Lhs, typename Rhs>
struct sub_expr {
    using gko_simd_expr = void;
    using tag = D;

    Lhs lhs;
    Rhs rhs;

    hn::Vec<D> eval(D d) const
    {
        if constexpr (is_mul_expr<Rhs>) {
            return hn::NegMulAdd(rhs.lhs.eval(d), rhs.rhs.eval(d), lhs.eval(d));
        } else {
            return hn::Sub(lhs.eval(d), rhs.eval(d));
        }
    }
};


#define GKO_SIMD_UNARY_NODE(_name, _op)                             \
    template <typename D, typename Arg>                             \
    struct _name {                                                  \
        using gko_simd_expr = void;                                 \
        using tag = D;                                              \
                                                                    \
        Arg arg;                                                    \
                                                                    \
        hn::Vec<D> eval(D d) const { return hn::_op(arg.eval(d)); } \
    }

GKO_SIMD_UNARY_NODE(neg_expr, Neg);
GKO_SIMD_UNARY_NODE(abs_expr, Abs);

#undef GKO_SIMD_UNARY_NODE


/**
 * @internal
 * squared_norm gets a dedicated node rather than being expressed as `e * e`,
 * so its operand is loaded once instead of twice.
 */
template <typename D, typename Arg>
struct squared_norm_expr {
    using gko_simd_expr = void;
    using tag = D;

    Arg arg;

    hn::Vec<D> eval(D d) const
    {
        const auto v = arg.eval(d);
        return hn::Mul(v, v);
    }
};


/** @internal Wraps a scalar in a leaf; passes an expression through. */
template <typename D, typename X>
auto as_expr(X x)
{
    if constexpr (is_expr<X>) {
        return x;
    } else {
        return scalar_leaf<D, X>{x};
    }
}


/**
 * @internal
 * Materializes an expression to a vector; passes a vector through.
 */
template <typename D, typename X>
hn::Vec<D> eval_to_vec(D d, X x)
{
    if constexpr (is_expr<X>) {
        return x.eval(d);
    } else {
        return x;
    }
}


/**
 * @internal
 * Mutable view of `count` contiguous lanes.
 * WARNING: a lazy alias into memory, not a value. Never bind one to `auto` and
 * read it after writing through the same accessor -- the deferred eval() would
 * re-read the overwritten memory.
 */
template <typename D, typename Access>
struct ref {
    using gko_simd_expr = void;
    using tag = D;

    hn::TFromD<D>* ptr;
    std::size_t count;

    hn::Vec<D> eval(D d) const
    {
        if constexpr (std::is_same_v<Access, partial_access>) {
            return hn::LoadN(d, ptr, count);
        } else {
            return hn::LoadU(d, ptr);
        }
    }

    void store(hn::Vec<D> v) const
    {
        if constexpr (std::is_same_v<Access, partial_access>) {
            hn::StoreN(v, D{}, ptr, count);
        } else {
            hn::StoreU(v, D{}, ptr);
        }
    }

    template <typename E>
    ref& operator=(E e)
    {
        store(as_expr<D>(e).eval(D{}));
        return *this;
    }

    // Written out: the implicit copy-assignment would copy the pointer, turning
    // `a(row, col) = b(row, col)` between two mutable accessors into a no-op.
    ref& operator=(const ref& other)
    {
        store(other.eval(D{}));
        return *this;
    }

    template <typename E>
    ref& operator+=(E e)
    {
        auto rhs = as_expr<D>(e);
        if constexpr (is_mul_expr<decltype(rhs)>) {
            store(hn::MulAdd(rhs.lhs.eval(D{}), rhs.rhs.eval(D{}), eval(D{})));
        } else {
            store(hn::Add(eval(D{}), rhs.eval(D{})));
        }
        return *this;
    }

    template <typename E>
    ref& operator-=(E e)
    {
        auto rhs = as_expr<D>(e);
        if constexpr (is_mul_expr<decltype(rhs)>) {
            store(
                hn::NegMulAdd(rhs.lhs.eval(D{}), rhs.rhs.eval(D{}), eval(D{})));
        } else {
            store(hn::Sub(eval(D{}), rhs.eval(D{})));
        }
        return *this;
    }

    template <typename E>
    ref& operator*=(E e)
    {
        store(hn::Mul(eval(D{}), as_expr<D>(e).eval(D{})));
        return *this;
    }

    template <typename E>
    ref& operator/=(E e)
    {
        store(hn::Div(eval(D{}), as_expr<D>(e).eval(D{})));
        return *this;
    }
};


/*
 * These build expression nodes and never call a Highway operator: Highway
 * defines none for SVE or RVV. Arithmetic goes through named ops in eval().
 */
#define GKO_SIMD_BINARY_OP(_sym, _node)                              \
    template <typename A, typename B,                                \
              typename = std::enable_if_t<is_expr<A> || is_expr<B>>> \
    auto operator _sym(A a, B b)                                     \
    {                                                                \
        using D = tag_of_t<A, B>;                                    \
        auto lhs = as_expr<D>(a);                                    \
        auto rhs = as_expr<D>(b);                                    \
        return _node<D, decltype(lhs), decltype(rhs)>{lhs, rhs};     \
    }

GKO_SIMD_BINARY_OP(+, add_expr)
GKO_SIMD_BINARY_OP(-, sub_expr)
GKO_SIMD_BINARY_OP(*, mul_expr)
GKO_SIMD_BINARY_OP(/, div_expr)

#undef GKO_SIMD_BINARY_OP


template <typename E, typename = std::enable_if_t<is_expr<E>>>
auto operator-(E e)
{
    using D = typename std::decay_t<E>::tag;
    return neg_expr<D, E>{e};
}


/*
 * `Node<Ts...>` rather than plain `E` so these win partial ordering against
 * `gko::zero`/`one`/`abs`/`conj`/`squared_norm`, which take `const T&` and are
 * also found by ordinary lookup from an unqualified call in a kernel body.
 * Requires every node above to have only type template parameters.
 */
template <template <typename...> class Node, typename... Ts,
          typename = std::enable_if_t<is_expr<Node<Ts...>>>>
auto zero(Node<Ts...>)
{
    using D = typename Node<Ts...>::tag;
    using T = hn::TFromD<D>;
    return scalar_leaf<D, T>{T{}};
}

template <template <typename...> class Node, typename... Ts,
          typename = std::enable_if_t<is_expr<Node<Ts...>>>>
auto one(Node<Ts...>)
{
    using D = typename Node<Ts...>::tag;
    using T = hn::TFromD<D>;
    return scalar_leaf<D, T>{T{1}};
}

template <template <typename...> class Node, typename... Ts,
          typename = std::enable_if_t<is_expr<Node<Ts...>>>>
auto abs(Node<Ts...> e)
{
    using D = typename Node<Ts...>::tag;
    return abs_expr<D, Node<Ts...>>{e};
}

/**
 * @internal
 * Real value types only in this iteration, so conj is the identity.
 */
template <template <typename...> class Node, typename... Ts,
          typename = std::enable_if_t<is_expr<Node<Ts...>>>>
auto conj(Node<Ts...> e)
{
    return e;
}

template <template <typename...> class Node, typename... Ts,
          typename = std::enable_if_t<is_expr<Node<Ts...>>>>
auto squared_norm(Node<Ts...> e)
{
    using D = typename Node<Ts...>::tag;
    return squared_norm_expr<D, Node<Ts...>>{e};
}


}  // namespace simd
}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_BASE_KERNEL_LAUNCH_SIMD_HPP_
