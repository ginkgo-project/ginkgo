// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_DIM_HPP_
#define GKO_PUBLIC_CORE_BASE_DIM_HPP_


#include <iostream>


#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * A type representing the dimensions of a multidimensional object.
 *
 * @tparam Dimensionality  number of dimensions of the object
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @ingroup dim
 */
template <size_type Dimensionality, typename DimensionType = size_type>
struct dim {
    static constexpr size_type dimensionality = Dimensionality;
    friend struct dim<dimensionality + 1>;

    using dimension_type = DimensionType;

    /**
     * Creates a dimension object with all dimensions set to zero.
     */
    constexpr GKO_ATTRIBUTES dim() : dim{dimension_type{}} {}

    /**
     * Creates a dimension object with all dimensions set to the same value.
     *
     * @param size  the size of each dimension
     */
    explicit constexpr GKO_ATTRIBUTES dim(const dimension_type& size)
        : first_{size}, rest_{size}
    {}

    /**
     * Creates a dimension object with the specified dimensions.
     *
     * If the number of dimensions given is less than the dimensionality of the
     * object, the remaining dimensions are set to the same value as the last
     * value given.
     *
     * For example, in the context of matrices `dim<2>{2, 3}` creates the
     * dimensions for a 2-by-3 matrix.
     *
     * @param first  first dimension
     * @param rest  other dimensions
     */
    template <typename... Rest, std::enable_if_t<sizeof...(Rest) ==
                                                 Dimensionality - 1>* = nullptr>
    constexpr GKO_ATTRIBUTES dim(const dimension_type& first,
                                 const Rest&... rest)
        : first_{first}, rest_{static_cast<dimension_type>(rest)...}
    {}

    /**
     * Returns the requested dimension.
     *
     * For example, if `d` is a dim<2> object representing matrix dimensions,
     * `d[0]` returns the number of rows, and `d[1]` returns the number of
     * columns.
     *
     * @param dimension  the requested dimension
     *
     * @return the `dimension`-th dimension
     */
    constexpr GKO_ATTRIBUTES const dimension_type& operator[](
        const size_type& dimension) const noexcept
    {
        return GKO_ASSERT(dimension < dimensionality),
               dimension == 0 ? first_ : rest_[dimension - 1];
    }

    /**
     * @copydoc operator[]() const
     */
    GKO_ATTRIBUTES dimension_type& operator[](
        const size_type& dimension) noexcept
    {
        return GKO_ASSERT(dimension < dimensionality),
               dimension == 0 ? first_ : rest_[dimension - 1];
    }

    /**
     * Checks if all dimensions evaluate to true.
     *
     * For standard arithmetic types, this is equivalent to all dimensions being
     * different than zero.
     *
     * @return true if and only if all dimensions evaluate to true
     *
     * @note This operator is explicit to avoid implicit dim-to-int casts.
     *       It will still be used in contextual conversions (if, &&, ||, !)
     */
    explicit constexpr GKO_ATTRIBUTES operator bool() const
    {
        return static_cast<bool>(first_) && static_cast<bool>(rest_);
    }

    /**
     * Checks if two dim objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend constexpr GKO_ATTRIBUTES bool operator==(const dim& x, const dim& y)
    {
        return x.first_ == y.first_ && x.rest_ == y.rest_;
    }

    /**
     * Checks if two dim objects are not equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return false if and only if all dimensions of both objects are equal.
     */
    friend constexpr GKO_ATTRIBUTES bool operator!=(const dim& x, const dim& y)
    {
        return !(x == y);
    }

    /**
     * Multiplies two dim objects.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return a dim object representing the size of the tensor product `x * y`
     */
    friend constexpr GKO_ATTRIBUTES dim operator*(const dim& x, const dim& y)
    {
        return dim(x.first_ * y.first_, x.rest_ * y.rest_);
    }

    /**
     * A stream operator overload for dim
     *
     * @param os  stream object
     * @param x  dim object
     *
     * @return a stream object appended with the dim output
     */
    friend std::ostream& operator<<(std::ostream& os, const dim& x)
    {
        os << "(";
        x.print_to(os);
        os << ")";
        return os;
    }

private:
    void inline print_to(std::ostream& os) const
    {
        os << first_ << ", ";
        rest_.print_to(os);
    }


    constexpr GKO_ATTRIBUTES dim(const dimension_type first,
                                 dim<dimensionality - 1> rest)
        : first_{first}, rest_{rest}
    {}

    dimension_type first_;
    dim<dimensionality - 1, dimension_type> rest_;
};


// base case for dim recursive template
template <typename DimensionType>
struct dim<1u, DimensionType> {
    static constexpr size_type dimensionality = 1u;
    friend struct dim<2>;

    using dimension_type = DimensionType;

    constexpr GKO_ATTRIBUTES dim(const dimension_type& size = dimension_type{})
        : first_{size}
    {}

    constexpr GKO_ATTRIBUTES const dimension_type& operator[](
        const size_type& dimension) const noexcept
    {
        return GKO_ASSERT(dimension == 0), first_;
    }

    GKO_ATTRIBUTES dimension_type& operator[](const size_type& dimension)
    {
        return GKO_ASSERT(dimension == 0), first_;
    }

    explicit constexpr GKO_ATTRIBUTES operator bool() const
    {
        return static_cast<bool>(first_);
    }

    friend constexpr GKO_ATTRIBUTES bool operator==(const dim& x, const dim& y)
    {
        return x.first_ == y.first_;
    }

    friend constexpr GKO_ATTRIBUTES bool operator!=(const dim& x, const dim& y)
    {
        return !(x == y);
    }

    friend constexpr GKO_ATTRIBUTES dim operator*(const dim& x, const dim& y)
    {
        return dim(x.first_ * y.first_);
    }

    friend std::ostream& operator<<(std::ostream& os, const dim& x)
    {
        os << "(";
        x.print_to(os);
        os << ")";
        return os;
    }

private:
    void inline print_to(std::ostream& os) const { os << first_; }

    dimension_type first_;
};


/**
 * Returns a dim<2> object with its dimensions swapped.
 *
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param dimensions original object
 *
 * @return a dim<2> object with its dimensions swapped
 */
template <typename DimensionType>
constexpr GKO_ATTRIBUTES GKO_INLINE dim<2, DimensionType> transpose(
    const dim<2, DimensionType>& dimensions) noexcept
{
    return {dimensions[1], dimensions[0]};
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DIM_HPP_
