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

#ifndef GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
#define GKO_CORE_BASE_EXTENDED_FLOAT_HPP_


#include <limits>
#include <type_traits>


#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef __CUDA_ARCH__


#include <cuda_fp16.h>


#elif defined(__HIP_DEVICE_COMPILE__)


#include <hip/hip_fp16.h>


#else
class __half;
#endif  // __CUDA_ARCH__


namespace gko {


/**
 * This template implements the truncated (or split) storage of a floating point
 * type.
 *
 * The class splits the type FloatType into NumComponents components. `i`-th
 * component of the class is represented by the instantiation with ComponentId
 * equal to `i`. 0-th component is the one with most significant bits, i.e. the
 * one that includes the sign and the exponent of the number.
 *
 * @tparam FloatType  a floating point type this truncates / splits
 * @tparam NumComponents  number of equally-sized components FloatType is split
 *                        into
 * @tparam ComponentId  index of the component of FloatType an object of this
 *                      template instntiation represents
 */
template <typename FloatType, size_type NumComponents,
          size_type ComponentId = 0>
class truncated {
public:
    using float_type = FloatType;

    /**
     * Unsigned type representing the bits of FloatType.
     */
    using full_bits_type = typename detail::float_traits<float_type>::bits_type;

    static constexpr auto num_components = NumComponents;
    static constexpr auto component_id = ComponentId;

    /**
     * Size of the component in bits.
     */
    static constexpr auto component_size =
        sizeof(float_type) * byte_size / num_components;
    /**
     * Starting bit position of the component in FloatType.
     */
    static constexpr auto component_position =
        (num_components - component_id - 1) * component_size;
    /**
     * Bitmask of the component in FloatType.
     */
    static constexpr auto component_mask =
        detail::create_ones<full_bits_type>(component_size)
        << component_position;

    /**
     * Unsigned type representing the bits of the component.
     */
    using bits_type = detail::uint_of<component_size>;

    static_assert((sizeof(float_type) * byte_size) % component_size == 0,
                  "Size of float is not a multiple of component size");
    static_assert(component_id < num_components,
                  "This type doesn't have that many components");

    truncated() noexcept = default;

    GKO_ATTRIBUTES explicit truncated(const float_type& val) noexcept
    {
        const auto& bits = reinterpret_cast<const full_bits_type&>(val);
        data_ = static_cast<bits_type>((bits & component_mask) >>
                                       component_position);
    }

    GKO_ATTRIBUTES operator float_type() const noexcept
    {
        const auto bits = static_cast<full_bits_type>(data_)
                          << component_position;
        return reinterpret_cast<const float_type&>(bits);
    }

    GKO_ATTRIBUTES truncated operator-() const noexcept
    {
        auto res = *this;
        // flip sign bit
        if (ComponentId == 0) {
            res.data_ ^= bits_type{1} << (8 * sizeof(bits_type) - 1);
        }
        return res;
    }

private:
    bits_type data_;
};


}  // namespace gko


namespace std {


template <typename T, gko::size_type NumComponents>
class complex<gko::truncated<T, NumComponents>> {
public:
    using value_type = gko::truncated<T, NumComponents>;

    complex(const value_type& real = 0.f, const value_type& imag = 0.f)
        : real_(real), imag_(imag)
    {}

    template <typename U>
    explicit complex(const complex<U>& other)
        : complex(static_cast<value_type>(other.real()),
                  static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }


    operator std::complex<T>() const noexcept
    {
        return std::complex<T>(static_cast<T>(real_), static_cast<T>(imag_));
    }

private:
    value_type real_;
    value_type imag_;
};


}  // namespace std


#endif  // GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
