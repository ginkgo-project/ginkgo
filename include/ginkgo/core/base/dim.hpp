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

#ifndef GKO_PUBLIC_CORE_BASE_DIM_HPP_
#define GKO_PUBLIC_CORE_BASE_DIM_HPP_


#include <iostream>
#include <vector>


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
     * Creates a dimension object with all dimensions set to the same value.
     *
     * @param size  the size of each dimension
     */
    constexpr GKO_ATTRIBUTES dim(const dimension_type& size = dimension_type{})
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
    template <typename... Rest>
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
     * Overload ostream operator to print the dimension to the output stream.
     */
    friend GKO_ATTRIBUTES std::ostream &operator<<(std::ostream &os,
                                                   const dim &out_dim)
    {
        os << " ( " << out_dim.first_ << " , " << out_dim.rest_ << " ) ";
        return os;
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


class batch_dim {
public:
    bool stores_equal_sizes() const { return equal_sizes_; }

    size_type get_num_batches() const { return num_batches_; }

    std::vector<dim<2>> get_batch_sizes() const
    {
        if (!equal_sizes_) {
            return sizes_;
        } else {
            return std::vector<dim<2>>(num_batches_, common_size_);
        }
    }

    const dim<2> &at(const size_type batch = 0) const
    {
        if (equal_sizes_) {
            return common_size_;
        } else {
            GKO_ASSERT(batch < num_batches_);
            return sizes_[batch];
        }
    }

    /**
     * Checks if two batch_dim objects are equal.
     *
     * @param x  first object
     * @param y  second object
     *
     * @return true if and only if all dimensions of both objects are equal.
     */
    friend const bool operator==(const batch_dim &x, const batch_dim &y)
    {
        if (x.equal_sizes_ && y.equal_sizes_) {
            return x.num_batches_ == y.num_batches_ &&
                   x.common_size_ == y.common_size_;
        } else {
            return x.sizes_ == y.sizes_;
        }
    }

    batch_dim(const size_type num_batches = 0, const dim<2> &size = dim<2>{})
        : equal_sizes_(true),
          common_size_(size),
          num_batches_(num_batches),
          sizes_()
    {}

    batch_dim(const std::vector<dim<2>> &batch_sizes)
        : equal_sizes_(false),
          common_size_(dim<2>{}),
          num_batches_(batch_sizes.size()),
          sizes_(batch_sizes)
    {
        check_size_equality();
    }

    void check_size_equality()
    {
        for (size_type i = 0; i < num_batches_; ++i) {
            if (!(sizes_[i] == sizes_[0])) {
                return;
            }
        }
        common_size_ = sizes_[0];
        equal_sizes_ = true;
    }

private:
    bool equal_sizes_{};
    size_type num_batches_{};
    dim<2> common_size_{};
    std::vector<dim<2>> sizes_{};
};


/**
 * Checks if two dim objects are different.
 *
 * @tparam Dimensionality  number of dimensions of the dim objects
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param x  first object
 * @param y  second object
 *
 * @return `!(x == y)`
 */
template <size_type Dimensionality, typename DimensionType>
constexpr GKO_ATTRIBUTES GKO_INLINE bool operator!=(
    const dim<Dimensionality, DimensionType>& x,
    const dim<Dimensionality, DimensionType>& y)
{
    return !(x == y);
}


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


/**
 * Returns a dim<2> object with its dimensions swapped for batched operators
 *
 * @tparam DimensionType  datatype used to represent each dimension
 *
 * @param dimensions original object
 *
 * @return a std::vector<dim<2>> object with its dimensions swapped
 */
inline batch_dim transpose(const batch_dim &input)
{
    batch_dim out{};
    if (input.stores_equal_sizes()) {
        out = batch_dim(input.get_num_batches(), gko::transpose(input.at(0)));
        return out;
    }
    auto trans = std::vector<dim<2>>(input.get_num_batches());
    for (size_type i = 0; i < trans.size(); ++i) {
        trans[i] = transpose(input.at(i));
    }
    return batch_dim(trans);
}


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_DIM_HPP_
