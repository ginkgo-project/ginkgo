/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_DIM_HPP_
#define GKO_CORE_BASE_DIM_HPP_


#include "core/base/types.hpp"
#include "core/base/utils.hpp"


namespace gko {


/**
 * A type representing the dimensions of a multidimensional object.
 *
 * @tparam Dimensionality  number of dimensions of the object
 * @tparam DimensionType  datatype used to represent each dimension
 */
template <size_type Dimensionality, typename DimensionType = size_type>
struct dim {
    static constexpr size_type dimensionality = Dimensionality;

    using dimension_type = DimensionType;

    /**
     * Creates a dimension object with all dimensions set to the same value.
     *
     * @param size  the size of each dimension
     */
    constexpr GKO_ATTRIBUTES dim(const dimension_type &size = dimension_type{})
        : first_{size}, rest_{size}
    {}

    /**
     * Creates a dimension object with the specified dimensions.
     *
     * If the number of dimensions given is less than the dimensionality of the
     * object, the remaining dimensions are set to the same value as the last
     * value given.
     *
     * @param first  first dimension
     * @param rest  other dimensions
     */
    template <typename... Rest>
    constexpr GKO_ATTRIBUTES dim(const dimension_type &first,
                                 const Rest &... rest)
        : first_{first}, rest_{static_cast<dimension_type>(rest)...}
    {}

    /**
     * Returns the requested dimension.
     *
     * @param dimension  the requested dimension
     *
     * @return the `dimension`-th dimension
     */
    constexpr GKO_ATTRIBUTES const dimension_type &operator[](
        const size_type &dimension) const noexcept
    {
        return GKO_ASSERT(dimension < dimensionality), *(&first_ + dimension);
    }

    /**
     * @copydoc operator[]() const
     */
    GKO_ATTRIBUTES dimension_type &operator[](
        const size_type &dimension) noexcept
    {
        return GKO_ASSERT(dimension < dimensionality), *(&first_ + dimension);
    }

    /**
     * Checks if all dimensions evaluate to true.
     *
     * For standard arithmetic types, this is equivalent to all dimensions being
     * different than zero.
     *
     * @return true if and only if all dimensions evaluate to true
     */
    constexpr GKO_ATTRIBUTES operator bool() const
    {
        return static_cast<bool>(first_) && static_cast<bool>(rest_);
    }

    /**
     * Checks if two dime objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend constexpr GKO_ATTRIBUTES bool operator==(const dim &x, const dim &y)
    {
        return x.first_ == y.first_ && x.rest_ == y.rest_;
    }

    /**
     * Multiplies two dim objects.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return a dim object representing the size of the tensor product `x * y`
     */
    friend constexpr GKO_ATTRIBUTES dim operator*(const dim &x, const dim &y)
    {
        return dim(x.first_ * y.first_, x.rest_ * y.rest_);
    }

private:
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

    using dimension_type = DimensionType;

    constexpr GKO_ATTRIBUTES dim(const dimension_type &size = dimension_type{})
        : first_{size}
    {}

    constexpr GKO_ATTRIBUTES const dimension_type &operator[](
        const size_type &dimension) const noexcept
    {
        return *(&first_ + dimension);
    }

    GKO_ATTRIBUTES dimension_type &operator[](const size_type &dimension)
    {
        return *(&first_ + dimension);
    }

    constexpr GKO_ATTRIBUTES operator bool() const
    {
        return static_cast<bool>(first_);
    }

    friend constexpr GKO_ATTRIBUTES bool operator==(const dim &x, const dim &y)
    {
        return x.first_ == y.first_;
    }

    friend constexpr GKO_ATTRIBUTES dim operator*(const dim &x, const dim &y)
    {
        return dim(x.first_ * y.first_);
    }

private:
    dimension_type first_;
};


/**
 * Checks if two dim objects are different.
 *
 * @param x  first object
 * @param y  second object
 *
 * @return `!(x == y)`
 */
template <size_type Dimensionality>
constexpr GKO_ATTRIBUTES GKO_INLINE bool operator!=(
    const dim<Dimensionality> &x, const dim<Dimensionality> &y)
{
    return !(x == y);
}


/**
 * Returns a dim<2> object with its dimensions swapped.
 *
 * @param dimensions original object
 *
 * @return a dim<2> object with its dimensions swapped
 */
constexpr GKO_ATTRIBUTES GKO_INLINE dim<2> transpose(
    const dim<2> &dimensions) noexcept
{
    return {dimensions[1], dimensions[0]};
}


}  // namespace gko


#endif  // GKO_CORE_BASE_DIM_HPP_
