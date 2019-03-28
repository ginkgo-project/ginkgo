/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_CORE_BASE_RANGE_HPP_
#define GKO_CORE_BASE_RANGE_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>


namespace gko {


/**
 * A span is a lightweight structure used to create sub-ranges from other
 * ranges.
 *
 * A `span s` represents a contiguous set of indexes in one dimension of the
 * range, starting on index `s.begin` (inclusive) and ending at index `s.end`
 * (exclusive). A span is only valid if its starting index is smaller than its
 * ending index.
 *
 * Spans can be compared using the `==` and `!=` operators. Two spans are
 * identical if both their `begin` and `end` values are identical.
 *
 * Spans also have two distinct partial orders defined on them:
 * 1.  `x < y` (`y > x`) if and only if `x.end < y.begin`
 * 2.  `x <= y` (`y >= x`) if and only if `x.end <= y.begin`
 *
 * Note that the orders are in fact partial - there are spans `x` and `y` for
 * which none of the following inequalities holds:
 *     `x < y`, `x > y`, `x == y`, `x <= y`, `x >= y`.
 * An example are spans `span{0, 2}` and `span{1, 3}`.
 *
 * In addition, `<=` is a distinct order from `<`, and not just an extension
 * of the strict order to its weak equivalent. Thus, `x <= y` is not equivalent
 * to `x < y || x == y`.
 */
struct span {
    /**
     * Creates a span representing a point `point`.
     *
     * The `begin` of this span is set to `point`, and the `end` to `point + 1`.
     *
     * @param point  the point which the span represents
     */
    GKO_ATTRIBUTES constexpr span(size_type point) noexcept
        : span{point, point + 1}
    {}

    /**
     * Creates a span.
     *
     * @param begin  the beginning of the span
     * @param end  the end of the span
     */
    GKO_ATTRIBUTES constexpr span(size_type begin, size_type end) noexcept
        : begin{begin}, end{end}
    {}

    /**
     * Checks if a span is valid.
     *
     * @return true if and only if `this->begin < this->end`
     */
    constexpr bool is_valid() const { return begin < end; }

    /**
     * Beginning of the span.
     */
    const size_type begin;

    /**
     * End of the span.
     */
    const size_type end;
};


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator<(const span &first,
                                                   const span &second)
{
    return first.end < second.begin;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator<=(const span &first,
                                                    const span &second)
{
    return first.end <= second.begin;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator>(const span &first,
                                                   const span &second)
{
    return second < first;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator>=(const span &first,
                                                    const span &second)
{
    return second <= first;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator==(const span &first,
                                                    const span &second)
{
    return first.begin == second.begin && first.end == second.end;
}


GKO_ATTRIBUTES GKO_INLINE constexpr bool operator!=(const span &first,
                                                    const span &second)
{
    return !(first == second);
}


namespace detail {


template <size_type CurrentDimension = 0, typename FirstRange,
          typename SecondRange>
GKO_ATTRIBUTES constexpr GKO_INLINE
    xstd::enable_if_t<(CurrentDimension >= max(FirstRange::dimensionality,
                                               SecondRange::dimensionality)),
                      bool>
    equal_dimensions(const FirstRange &, const SecondRange &)
{
    return true;
}

template <size_type CurrentDimension = 0, typename FirstRange,
          typename SecondRange>
GKO_ATTRIBUTES constexpr GKO_INLINE
    xstd::enable_if_t<(CurrentDimension < max(FirstRange::dimensionality,
                                              SecondRange::dimensionality)),
                      bool>
    equal_dimensions(const FirstRange &first, const SecondRange &second)
{
    return first.length(CurrentDimension) == second.length(CurrentDimension) &&
           equal_dimensions<CurrentDimension + 1>(first, second);
}


}  // namespace detail


/**
 * A range is a multidimensional view of the memory.
 *
 * The range does not store any of its values by itself. Instead, it obtains the
 * values through an accessor (e.g. accessor::row_major) which describes how the
 * indexes of the range map to physical locations in memory.
 *
 * There are several advantages of using ranges instead of plain memory
 * pointers:
 *
 * 1.  Code using ranges is easier to read and write, as there is no need for
 *     index linearizations.
 * 2.  Code using ranges is safer, as it is impossible to accidentally
 *     miscalculate an index or step out of bounds, since range accessors
 *     perform bounds checking in debug builds. For performance, this can be
 *     disabled in release builds by defining the `NDEBUG` flag.
 * 3.  Ranges enable generalized code, as algorithms can be written independent
 *     of the memory layout. This does not impede various optimizations based
 *     on memory layout, as it is always possible to specialize algorithms for
 *     ranges with specific memory layouts.
 * 4.  Ranges have various pointwise operations predefined, which reduces the
 *     amount of loops that need to be written.
 *
 * Range operations
 * ----------------
 *
 * Ranges define a complete set of pointwise unary and binary operators which
 * extend the basic arithmetic operators in C++, as well as a few pointwise
 * operations and mathematical functions useful in ginkgo, and a couple of
 * non-pointwise operations. Compound assignment (`+=`, `*=`, etc.) is not yet
 * supported at this moment. Here is a complete list of operations:
 *
 * -   standard unary operations: `+`, `-`, `!`, `~`
 * -   standard binary operations: `+`, `*` (this is pointwise, not
 *     matrix multiplication), `/`, `%`, `<`, `>`, `<=`, `>=`, `==`, `!=`,
 *     `||`, `&&`, `|`, `&`, `^`, `<<`, `>>`
 * -   useful unary functions: `zero`, `one`, `abs`, `real`, `imag`, `conj`,
 *                             `squared_norm`
 * -   useful binary functions: `min`, `max`
 *
 * All binary pointwise operations also work as expected if one of the operands
 * is a scalar and the other is a range. The scalar operand will have the effect
 * as if it was a range of the same size as the other operand, filled with
 * the specified scalar.
 *
 * Two "global" functions `transpose` and `mmul` are also supported.
 * `transpose` transposes the first two dimensions of the range (i.e.
 * `transpose(r)(i, j, ...) == r(j, i, ...)`).
 * `mmul` performs a (batched) matrix multiply of the ranges - the first two
 * dimensions represent the matrices, while the rest represent the batch.
 * For example, given the ranges `r1` and `r2` of dimensions `(3, 2, 3)` and
 * `(2, 4, 3)`, respectively, `mmul(r1, r2)` will return a range of dimensions
 * `(3, 4, 3)`, obtained by multiplying the 3 frontal slices of the range, and
 * stacking the result back vertically.
 *
 * Compound operations
 * -------------------
 *
 * Multiple range operations can be combined into a single expression. For
 * example, an "axpy" operation can be obtained using `y = alpha * x + y`, where
 * `x` an `y` are ranges, and `alpha` is a scalar.
 * Range operations are optimized for memory access, and the above code does not
 * allocate additional storage for intermediate ranges `alpha * x`
 * or `aplha * x + y`. In fact, the entire computation is done during the
 * assignment, and the results of operations `+` and `*` only register the data,
 * and the types of operations that will be computed once the results are
 * needed.
 *
 * It is possible to store and reuse these intermediate expressions. The
 * following example will overwrite the range `x` with it's 4th power:
 *
 * ```c++
 * auto square = x * x;  // this is range constructor, not range assignment!
 * x = square;  // overwrites x with x*x (this is range assignment)
 * x = square;  // overwrites new x (x*x) with (x*x)*(x*x) (as is this)
 * ```
 *
 * Caveats
 * -------
 *
 * __`mmul` is not a highly-optimized BLAS-3 version of the matrix
 * multiplication.__ The current design of ranges and accessors prevents that,
 * so if you need a high-perfromance matrix multiplication, you should use one
 * of the libraries that provide that, or implement your own
 * (you can use pointwise range operations to help simplify that). However,
 * range design might get improved in the future to allow efficient
 * implementations of BLAS-3 kernels.
 *
 * Aliasing the result range in `mmul` and `transpose` is not allowed.
 * Constructs like `A = transpose(A)`, `A = mmul(A, A)`, or `A = mmul(A, A) + C`
 * lead to undefined behavior.
 * However, aliasing input arguments is allowed: `C = mmul(A, A)`, and even
 * `C = mmul(A, A) + C` is valid code (in the last example, only pointwise
 * operations are aliased). `C = mmul(A, A + C)` is not valid though.
 *
 * Examples
 * --------
 *
 * The range unit tests in core/test/base/range.cpp contain lots of simple
 * 1-line examples of range operations. The accessor unit tests in
 * core/test/base/range.cpp show how to use ranges with concrete accessors,
 * and how to use range slices using `span`s as arguments to range function call
 * operator. Finally, examples/range contains a complete example where ranges
 * are used to implement a simple version of the right-looking LU factorization.
 *
 * @tparam Accessor  underlying accessor of the range
 */
template <typename Accessor>
class range {
public:
    /**
     * The type of the underlying accessor.
     */
    using accessor = Accessor;

    /**
     * The number of dimensions of the range.
     */
    static constexpr size_type dimensionality = accessor::dimensionality;

    /**
     * Creates a new range.
     *
     * @tparam AccessorParam  types of parameters forwarded to the accessor
     *                        constructor
     *
     * @param params  parameters forwarded to Accessor constructor.
     */
    template <typename... AccessorParams>
    GKO_ATTRIBUTES constexpr explicit range(AccessorParams &&... params)
        : accessor_{std::forward<AccessorParams>(params)...}
    {}

    /**
     * Returns a value (or a sub-range) with the specified indexes.
     *
     * @tparam DimensionTypes  The types of indexes. Supported types depend on
     *                         the underlying accessor, but are usually either
     *                         integer types or spans. If at least one index is
     *                         a span, the returned value will be a sub-range.
     *
     * @param dimensions  the indexes of the values.
     *
     * @return a value on position `(dimensions...)`.
     */
    template <typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(DimensionTypes &&... dimensions)
        const -> decltype(std::declval<accessor>()(
            std::forward<DimensionTypes>(dimensions)...))
    {
        static_assert(sizeof...(dimensions) <= dimensionality,
                      "Too many dimensions in range call");
        return accessor_(std::forward<DimensionTypes>(dimensions)...);
    }

    /**
     * @copydoc operator=(const range &other)
     *
     * This is a version of the function which allows to copy between ranges
     * of different accessors.
     *
     * @tparam OtherAccessor  accessor of the other range
     */
    template <typename OtherAccessor>
    GKO_ATTRIBUTES const range &operator=(
        const range<OtherAccessor> &other) const
    {
        GKO_ASSERT(detail::equal_dimensions(*this, other));
        accessor_.copy_from(other);
        return *this;
    }

    /**
     * Assigns another range to this range.
     *
     * The order of assignment is defined by the accessor of this range, thus
     * the memory access will be optimized for the resulting range, and not for
     * the other range. If the sizes of two ranges do not match, the result is
     * undefined. Sizes of the ranges are checked at runtime in debug builds.
     *
     * @note Temporary accessors are allowed to define the implementation of
     *       the assignment as deleted, so do not expect `r1 * r2 = r2` to work.
     *
     * @param other  the range to copy the data from
     */
    GKO_ATTRIBUTES const range &operator=(const range &other) const
    {
        GKO_ASSERT(detail::equal_dimensions(*this, other));
        accessor_.copy_from(other.get_accessor());
        return *this;
    }

    /**
     * Returns the length of the specified dimension of the range.
     *
     * @param dimension  the dimensions whose length is returned
     *
     * @return  the length of the `dimension`-th dimension of the range
     */
    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return accessor_.length(dimension);
    }

    /**
     * Returns a pointer to the accessor.
     *
     * Can be used to access data and functions of a specific accessor.
     *
     * @return pointer to the accessor
     */
    GKO_ATTRIBUTES constexpr const accessor *operator->() const noexcept
    {
        return &accessor_;
    }

    /**
     * `Returns a reference to the accessor.
     *
     * @return reference to the accessor
     */
    GKO_ATTRIBUTES constexpr const accessor &get_accessor() const noexcept
    {
        return accessor_;
    }

private:
    accessor accessor_;
};


// implementation of range operations follows
// (you probably should not have to look at this unless you're interested in the
// gory details)


namespace detail {


enum class operation_kind { range_by_range, scalar_by_range, range_by_scalar };


template <typename Accessor, typename Operation>
struct implement_unary_operation {
    using accessor = Accessor;
    static constexpr size_type dimensionality = accessor::dimensionality;

    GKO_ATTRIBUTES constexpr explicit implement_unary_operation(
        const Accessor &operand)
        : operand{operand}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(
        const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate(std::declval<accessor>(),
                                        dimensions...))
    {
        return Operation::evaluate(operand, dimensions...);
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return operand.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const accessor operand;
};


template <operation_kind Kind, typename FirstOperand, typename SecondOperand,
          typename Operation>
struct implement_binary_operation {};

template <typename FirstAccessor, typename SecondAccessor, typename Operation>
struct implement_binary_operation<operation_kind::range_by_range, FirstAccessor,
                                  SecondAccessor, Operation> {
    using first_accessor = FirstAccessor;
    using second_accessor = SecondAccessor;
    static_assert(first_accessor::dimensionality ==
                      second_accessor::dimensionality,
                  "Both ranges need to have the same number of dimensions");
    static constexpr size_type dimensionality = first_accessor::dimensionality;

    GKO_ATTRIBUTES explicit implement_binary_operation(
        const FirstAccessor &first, const SecondAccessor &second)
        : first{first}, second{second}
    {
        GKO_ASSERT(gko::detail::equal_dimensions(first, second));
    }

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(
        const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_range_by_range(
            std::declval<first_accessor>(), std::declval<second_accessor>(),
            dimensions...))
    {
        return Operation::evaluate_range_by_range(first, second, dimensions...);
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return first.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const first_accessor first;
    const second_accessor second;
};

template <typename FirstOperand, typename SecondAccessor, typename Operation>
struct implement_binary_operation<operation_kind::scalar_by_range, FirstOperand,
                                  SecondAccessor, Operation> {
    using second_accessor = SecondAccessor;
    static constexpr size_type dimensionality = second_accessor::dimensionality;

    GKO_ATTRIBUTES constexpr explicit implement_binary_operation(
        const FirstOperand &first, const SecondAccessor &second)
        : first{first}, second{second}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(
        const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_scalar_by_range(
            std::declval<FirstOperand>(), std::declval<second_accessor>(),
            dimensions...))
    {
        return Operation::evaluate_scalar_by_range(first, second,
                                                   dimensions...);
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return second.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const FirstOperand first;
    const second_accessor second;
};

template <typename FirstAccessor, typename SecondOperand, typename Operation>
struct implement_binary_operation<operation_kind::range_by_scalar,
                                  FirstAccessor, SecondOperand, Operation> {
    using first_accessor = FirstAccessor;
    static constexpr size_type dimensionality = first_accessor::dimensionality;

    GKO_ATTRIBUTES constexpr explicit implement_binary_operation(
        const FirstAccessor &first, const SecondOperand &second)
        : first{first}, second{second}
    {}

    template <typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(
        const DimensionTypes &... dimensions) const
        -> decltype(Operation::evaluate_range_by_scalar(
            std::declval<first_accessor>(), std::declval<SecondOperand>(),
            dimensions...))
    {
        return Operation::evaluate_range_by_scalar(first, second,
                                                   dimensions...);
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return first.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const first_accessor first;
    const SecondOperand second;
};


}  // namespace detail


#define GKO_ENABLE_UNARY_RANGE_OPERATION(_operation_name, _operator_name, \
                                         _operator)                       \
    namespace accessor {                                                  \
    template <typename Operand>                                           \
    struct _operation_name                                                \
        : ::gko::detail::implement_unary_operation<Operand,               \
                                                   ::gko::_operator> {    \
        using ::gko::detail::implement_unary_operation<                   \
            Operand, ::gko::_operator>::implement_unary_operation;        \
    };                                                                    \
    }                                                                     \
    GKO_BIND_UNARY_RANGE_OPERATION_TO_OPERATOR(_operation_name, _operator_name)


#define GKO_BIND_UNARY_RANGE_OPERATION_TO_OPERATOR(_operation_name,          \
                                                   _operator_name)           \
    template <typename Accessor>                                             \
    GKO_ATTRIBUTES constexpr GKO_INLINE                                      \
        range<accessor::_operation_name<Accessor>>                           \
        _operator_name(const range<Accessor> &operand)                       \
    {                                                                        \
        return range<accessor::_operation_name<Accessor>>(                   \
            operand.get_accessor());                                         \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define GKO_DEFINE_SIMPLE_UNARY_OPERATION(_name, ...)                  \
    struct _name {                                                     \
    private:                                                           \
        template <typename Operand>                                    \
        GKO_ATTRIBUTES static constexpr auto simple_evaluate_impl(     \
            const Operand &operand) -> decltype(__VA_ARGS__)           \
        {                                                              \
            return __VA_ARGS__;                                        \
        }                                                              \
                                                                       \
    public:                                                            \
        template <typename AccessorType, typename... DimensionTypes>   \
        GKO_ATTRIBUTES static constexpr auto evaluate(                 \
            const AccessorType &accessor,                              \
            const DimensionTypes &... dimensions)                      \
            -> decltype(simple_evaluate_impl(accessor(dimensions...))) \
        {                                                              \
            return simple_evaluate_impl(accessor(dimensions...));      \
        }                                                              \
    }


namespace accessor {
namespace detail {


// unary arithmetic
GKO_DEFINE_SIMPLE_UNARY_OPERATION(unary_plus, +operand);
GKO_DEFINE_SIMPLE_UNARY_OPERATION(unary_minus, -operand);

// unary logical
GKO_DEFINE_SIMPLE_UNARY_OPERATION(logical_not, !operand);

// unary bitwise
GKO_DEFINE_SIMPLE_UNARY_OPERATION(bitwise_not, ~(operand));

// common functions
GKO_DEFINE_SIMPLE_UNARY_OPERATION(zero_operation, zero(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(one_operation, one(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(abs_operation, abs(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(real_operation, real(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(imag_operation, imag(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(conj_operation, conj(operand));
GKO_DEFINE_SIMPLE_UNARY_OPERATION(squared_norm_operation,
                                  squared_norm(operand));

}  // namespace detail
}  // namespace accessor


// unary arithmetic
GKO_ENABLE_UNARY_RANGE_OPERATION(unary_plus, operator+,
                                 accessor::detail::unary_plus);
GKO_ENABLE_UNARY_RANGE_OPERATION(unary_minus, operator-,
                                 accessor::detail::unary_minus);

// unary logical
GKO_ENABLE_UNARY_RANGE_OPERATION(logical_not, operator!,
                                 accessor::detail::logical_not);

// unary bitwise
GKO_ENABLE_UNARY_RANGE_OPERATION(bitwise_not, operator~,
                                 accessor::detail::bitwise_not);

// common unary functions
GKO_ENABLE_UNARY_RANGE_OPERATION(zero_operation, zero,
                                 accessor::detail::zero_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(one_operaton, one,
                                 accessor::detail::one_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(abs_operaton, abs,
                                 accessor::detail::abs_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(real_operaton, real,
                                 accessor::detail::real_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(imag_operaton, imag,
                                 accessor::detail::imag_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(conj_operaton, conj,
                                 accessor::detail::conj_operation);
GKO_ENABLE_UNARY_RANGE_OPERATION(squared_norm_operaton, squared_norm,
                                 accessor::detail::squared_norm_operation);

namespace accessor {


template <typename Accessor>
struct transpose_operation {
    using accessor = Accessor;
    static constexpr size_type dimensionality = accessor::dimensionality;

    GKO_ATTRIBUTES constexpr explicit transpose_operation(
        const Accessor &operand)
        : operand{operand}
    {}

    template <typename FirstDimensionType, typename SecondDimensionType,
              typename... DimensionTypes>
    GKO_ATTRIBUTES constexpr auto operator()(
        const FirstDimensionType &first_dim,
        const SecondDimensionType &second_dim,
        const DimensionTypes &... dims) const
        -> decltype(std::declval<accessor>()(second_dim, first_dim, dims...))
    {
        return operand(second_dim, first_dim, dims...);
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return dimension < 2 ? operand.length(dimension ^ 1)
                             : operand.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const accessor operand;
};


}  // namespace accessor


GKO_BIND_UNARY_RANGE_OPERATION_TO_OPERATOR(transpose_operation, transpose);


#undef GKO_DEFINE_SIMPLE_UNARY_OPERATION
#undef GKO_ENABLE_UNARY_RANGE_OPERATION


#define GKO_ENABLE_BINARY_RANGE_OPERATION(_operation_name, _operator_name,   \
                                          _operator)                         \
    namespace accessor {                                                     \
    template <::gko::detail::operation_kind Kind, typename FirstOperand,     \
              typename SecondOperand>                                        \
    struct _operation_name                                                   \
        : ::gko::detail::implement_binary_operation<                         \
              Kind, FirstOperand, SecondOperand, ::gko::_operator> {         \
        using ::gko::detail::implement_binary_operation<                     \
            Kind, FirstOperand, SecondOperand,                               \
            ::gko::_operator>::implement_binary_operation;                   \
    };                                                                       \
    }                                                                        \
    GKO_BIND_RANGE_OPERATION_TO_OPERATOR(_operation_name, _operator_name);   \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define GKO_BIND_RANGE_OPERATION_TO_OPERATOR(_operation_name, _operator_name) \
    template <typename Accessor>                                              \
    GKO_ATTRIBUTES constexpr GKO_INLINE range<accessor::_operation_name<      \
        ::gko::detail::operation_kind::range_by_range, Accessor, Accessor>>   \
    _operator_name(const range<Accessor> &first,                              \
                   const range<Accessor> &second)                             \
    {                                                                         \
        return range<accessor::_operation_name<                               \
            ::gko::detail::operation_kind::range_by_range, Accessor,          \
            Accessor>>(first.get_accessor(), second.get_accessor());          \
    }                                                                         \
                                                                              \
    template <typename FirstAccessor, typename SecondAccessor>                \
    GKO_ATTRIBUTES constexpr GKO_INLINE range<accessor::_operation_name<      \
        ::gko::detail::operation_kind::range_by_range, FirstAccessor,         \
        SecondAccessor>>                                                      \
    _operator_name(const range<FirstAccessor> &first,                         \
                   const range<SecondAccessor> &second)                       \
    {                                                                         \
        return range<accessor::_operation_name<                               \
            ::gko::detail::operation_kind::range_by_range, FirstAccessor,     \
            SecondAccessor>>(first.get_accessor(), second.get_accessor());    \
    }                                                                         \
                                                                              \
    template <typename FirstAccessor, typename SecondOperand>                 \
    GKO_ATTRIBUTES constexpr GKO_INLINE range<accessor::_operation_name<      \
        ::gko::detail::operation_kind::range_by_scalar, FirstAccessor,        \
        SecondOperand>>                                                       \
    _operator_name(const range<FirstAccessor> &first,                         \
                   const SecondOperand &second)                               \
    {                                                                         \
        return range<accessor::_operation_name<                               \
            ::gko::detail::operation_kind::range_by_scalar, FirstAccessor,    \
            SecondOperand>>(first.get_accessor(), second);                    \
    }                                                                         \
                                                                              \
    template <typename FirstOperand, typename SecondAccessor>                 \
    GKO_ATTRIBUTES constexpr GKO_INLINE range<accessor::_operation_name<      \
        ::gko::detail::operation_kind::scalar_by_range, FirstOperand,         \
        SecondAccessor>>                                                      \
    _operator_name(const FirstOperand &first,                                 \
                   const range<SecondAccessor> &second)                       \
    {                                                                         \
        return range<accessor::_operation_name<                               \
            ::gko::detail::operation_kind::scalar_by_range, FirstOperand,     \
            SecondAccessor>>(first, second.get_accessor());                   \
    }                                                                         \
    static_assert(true,                                                       \
                  "This assert is used to counter the false positive extra "  \
                  "semi-colon warnings")


#define GKO_DEFINE_SIMPLE_BINARY_OPERATION(_name, ...)                         \
    struct _name {                                                             \
    private:                                                                   \
        template <typename FirstOperand, typename SecondOperand>               \
        GKO_ATTRIBUTES constexpr static auto simple_evaluate_impl(             \
            const FirstOperand &first, const SecondOperand &second)            \
            -> decltype(__VA_ARGS__)                                           \
        {                                                                      \
            return __VA_ARGS__;                                                \
        }                                                                      \
                                                                               \
    public:                                                                    \
        template <typename FirstAccessor, typename SecondAccessor,             \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static constexpr auto evaluate_range_by_range(          \
            const FirstAccessor &first, const SecondAccessor &second,          \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first(dims...), second(dims...))) \
        {                                                                      \
            return simple_evaluate_impl(first(dims...), second(dims...));      \
        }                                                                      \
                                                                               \
        template <typename FirstOperand, typename SecondAccessor,              \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static constexpr auto evaluate_scalar_by_range(         \
            const FirstOperand &first, const SecondAccessor &second,           \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first, second(dims...)))          \
        {                                                                      \
            return simple_evaluate_impl(first, second(dims...));               \
        }                                                                      \
                                                                               \
        template <typename FirstAccessor, typename SecondOperand,              \
                  typename... DimensionTypes>                                  \
        GKO_ATTRIBUTES static constexpr auto evaluate_range_by_scalar(         \
            const FirstAccessor &first, const SecondOperand &second,           \
            const DimensionTypes &... dims)                                    \
            -> decltype(simple_evaluate_impl(first(dims...), second))          \
        {                                                                      \
            return simple_evaluate_impl(first(dims...), second);               \
        }                                                                      \
    }


namespace accessor {
namespace detail {


// binary arithmetic
GKO_DEFINE_SIMPLE_BINARY_OPERATION(add, first + second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(sub, first - second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(mul, first *second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(div, first / second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(mod, first % second);

// relational
GKO_DEFINE_SIMPLE_BINARY_OPERATION(less, first < second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(greater, first > second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(less_or_equal, first <= second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(greater_or_equal, first >= second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(equal, first == second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(not_equal, first != second);

// binary logical
GKO_DEFINE_SIMPLE_BINARY_OPERATION(logical_or, first || second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(logical_and, first &&second);

// binary bitwise
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_or, first | second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_and, first &second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(bitwise_xor, first ^ second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(left_shift, first << second);
GKO_DEFINE_SIMPLE_BINARY_OPERATION(right_shift, first >> second);

// common binary functions
GKO_DEFINE_SIMPLE_BINARY_OPERATION(max_operation, max(first, second));
GKO_DEFINE_SIMPLE_BINARY_OPERATION(min_operation, min(first, second));

}  // namespace detail
}  // namespace accessor


// binary arithmetic
GKO_ENABLE_BINARY_RANGE_OPERATION(add, operator+, accessor::detail::add);
GKO_ENABLE_BINARY_RANGE_OPERATION(sub, operator-, accessor::detail::sub);
GKO_ENABLE_BINARY_RANGE_OPERATION(mul, operator*, accessor::detail::mul);
GKO_ENABLE_BINARY_RANGE_OPERATION(div, operator/, accessor::detail::div);
GKO_ENABLE_BINARY_RANGE_OPERATION(mod, operator%, accessor::detail::mod);

// relational
GKO_ENABLE_BINARY_RANGE_OPERATION(less, operator<, accessor::detail::less);
GKO_ENABLE_BINARY_RANGE_OPERATION(greater, operator>,
                                  accessor::detail::greater);
GKO_ENABLE_BINARY_RANGE_OPERATION(less_or_equal, operator<=,
                                  accessor::detail::less_or_equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(greater_or_equal, operator>=,
                                  accessor::detail::greater_or_equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(equal, operator==, accessor::detail::equal);
GKO_ENABLE_BINARY_RANGE_OPERATION(not_equal, operator!=,
                                  accessor::detail::not_equal);

// binary logical
GKO_ENABLE_BINARY_RANGE_OPERATION(logical_or, operator||,
                                  accessor::detail::logical_or);
GKO_ENABLE_BINARY_RANGE_OPERATION(logical_and, operator&&,
                                  accessor::detail::logical_and);

// binary bitwise
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_or, operator|,
                                  accessor::detail::bitwise_or);
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_and, operator&,
                                  accessor::detail::bitwise_and);
GKO_ENABLE_BINARY_RANGE_OPERATION(bitwise_xor, operator^,
                                  accessor::detail::bitwise_xor);
GKO_ENABLE_BINARY_RANGE_OPERATION(left_shift, operator<<,
                                  accessor::detail::left_shift);
GKO_ENABLE_BINARY_RANGE_OPERATION(right_shift, operator>>,
                                  accessor::detail::right_shift);

// common binary functions
GKO_ENABLE_BINARY_RANGE_OPERATION(max_operaton, max,
                                  accessor::detail::max_operation);
GKO_ENABLE_BINARY_RANGE_OPERATION(min_operaton, min,
                                  accessor::detail::min_operation);


// special binary range functions
namespace accessor {


template <gko::detail::operation_kind Kind, typename FirstAccessor,
          typename SecondAccessor>
struct mmul_operation {
    static_assert(Kind == gko::detail::operation_kind::range_by_range,
                  "Matrix multiplication expects both operands to be ranges");
    using first_accessor = FirstAccessor;
    using second_accessor = SecondAccessor;
    static_assert(first_accessor::dimensionality ==
                      second_accessor::dimensionality,
                  "Both ranges need to have the same number of dimensions");
    static constexpr size_type dimensionality = first_accessor::dimensionality;

    GKO_ATTRIBUTES explicit mmul_operation(const FirstAccessor &first,
                                           const SecondAccessor &second)
        : first{first}, second{second}
    {
        GKO_ASSERT(first.length(1) == second.length(0));
        GKO_ASSERT(gko::detail::equal_dimensions<2>(first, second));
    }

    template <typename FirstDimension, typename SecondDimension,
              typename... DimensionTypes>
    GKO_ATTRIBUTES auto operator()(const FirstDimension &row,
                                   const SecondDimension &col,
                                   const DimensionTypes &... rest) const
        -> decltype(std::declval<FirstAccessor>()(row, 0, rest...) *
                        std::declval<SecondAccessor>()(0, col, rest...) +
                    std::declval<FirstAccessor>()(row, 1, rest...) *
                        std::declval<SecondAccessor>()(1, col, rest...))
    {
        using result_type =
            decltype(first(row, 0, rest...) * second(0, col, rest...) +
                     first(row, 1, rest...) * second(1, col, rest...));
        GKO_ASSERT(first.length(1) == second.length(0));
        auto result = zero<result_type>();
        const auto size = first.length(1);
        for (auto i = zero(size); i < size; ++i) {
            result += first(row, i, rest...) * second(i, col, rest...);
        }
        return result;
    }

    GKO_ATTRIBUTES constexpr size_type length(size_type dimension) const
    {
        return dimension == 1 ? second.length(1) : first.length(dimension);
    }

    template <typename OtherAccessor>
    GKO_ATTRIBUTES void copy_from(const OtherAccessor &other) const = delete;

    const first_accessor first;
    const second_accessor second;
};


}  // namespace accessor


GKO_BIND_RANGE_OPERATION_TO_OPERATOR(mmul_operation, mmul);


#undef GKO_DEFINE_SIMPLE_BINARY_OPERATION
#undef GKO_ENABLE_BINARY_RANGE_OPERATION


}  // namespace gko


#endif  // GKO_CORE_BASE_RANGE_HPP_
