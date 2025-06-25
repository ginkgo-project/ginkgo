// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_COMPLEX_HPP_
#define GKO_DPCPP_BASE_COMPLEX_HPP_

#include <complex>

#include <sycl/half_type.hpp>

#include <ginkgo/config.hpp>

#include "dpcpp/base/bf16_alias.hpp"


namespace gko {

template <typename>
class complex;


template <>
class complex<sycl::half> {
public:
    using value_type = sycl::half;

    complex(const value_type& real = value_type(0.f),
            const value_type& imag = value_type(0.f))
        : real_(real), imag_(imag)
    {}

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_scalar<T>::value &&
                                          std::is_scalar<U>::value>>
    explicit complex(const T& real, const U& imag)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(imag))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const T& real)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(0.f))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const std::complex<T>& other)
        : real_(static_cast<value_type>(other.real())),
          imag_(static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }

    operator std::complex<float>() const noexcept
    {
        return std::complex<float>(static_cast<float>(real_),
                                   static_cast<float>(imag_));
    }

    bool operator!=(const complex& r) const { return !this->operator==(r); }

    bool operator==(const complex& r) const
    {
        return real_ == r.real() && imag_ == r.imag();
    }

    template <typename V>
    complex& operator=(const V& val)
    {
        real_ = val;
        imag_ = value_type();
        return *this;
    }

    template <typename V>
    complex& operator=(const std::complex<V>& val)
    {
        real_ = val.real();
        imag_ = val.imag();
        return *this;
    }

    complex& operator+=(const value_type& real)
    {
        real_ += real;
        return *this;
    }

    complex& operator-=(const value_type& real)
    {
        real_ -= real;
        return *this;
    }

    complex& operator*=(const value_type& real)
    {
        real_ *= real;
        imag_ *= real;
        return *this;
    }

    complex& operator/=(const value_type& real)
    {
        real_ /= real;
        imag_ /= real;
        return *this;
    }

    template <typename T>
    complex& operator+=(const std::complex<T>& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    template <typename T>
    complex& operator-=(const std::complex<T>& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    template <typename T>
    complex& operator*=(const std::complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    template <typename T>
    complex& operator/=(const std::complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    complex& operator+=(const complex& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    complex& operator-=(const complex& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    complex& operator*=(const complex& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    complex& operator/=(const complex& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

#define COMPLEX_HALF_OPERATOR(_op, _opeq)                               \
    friend complex operator _op(const complex& lhf, const complex& rhf) \
    {                                                                   \
        auto a = lhf;                                                   \
        a _opeq rhf;                                                    \
        return a;                                                       \
    }

    COMPLEX_HALF_OPERATOR(+, +=)
    COMPLEX_HALF_OPERATOR(-, -=)
    COMPLEX_HALF_OPERATOR(*, *=)
    COMPLEX_HALF_OPERATOR(/, /=)

#undef COMPLEX_HALF_OPERATOR

    complex operator-() const { return complex(-real_, -imag_); }

private:
    value_type real_;
    value_type imag_;
};


template <>
class complex<vendor_bf16> {
public:
    using value_type = vendor_bf16;

    complex(const value_type& real = value_type(0.f),
            const value_type& imag = value_type(0.f))
        : real_(real), imag_(imag)
    {}

    template <typename T, typename U,
              typename = std::enable_if_t<std::is_scalar<T>::value &&
                                          std::is_scalar<U>::value>>
    explicit complex(const T& real, const U& imag)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(imag))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const T& real)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(0.f))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const std::complex<T>& other)
        : real_(static_cast<value_type>(other.real())),
          imag_(static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }

    operator std::complex<float>() const noexcept
    {
        return std::complex<float>(static_cast<float>(real_),
                                   static_cast<float>(imag_));
    }

    bool operator!=(const complex& r) const { return !this->operator==(r); }

    bool operator==(const complex& r) const
    {
        return real_ == r.real() && imag_ == r.imag();
    }

    template <typename V>
    complex& operator=(const V& val)
    {
        real_ = val;
        imag_ = value_type();
        return *this;
    }

    template <typename V>
    complex& operator=(const std::complex<V>& val)
    {
        real_ = val.real();
        imag_ = val.imag();
        return *this;
    }

    complex& operator+=(const value_type& real)
    {
        real_ += real;
        return *this;
    }

    complex& operator-=(const value_type& real)
    {
        real_ -= real;
        return *this;
    }

    complex& operator*=(const value_type& real)
    {
        real_ *= real;
        imag_ *= real;
        return *this;
    }

    complex& operator/=(const value_type& real)
    {
        real_ /= real;
        imag_ /= real;
        return *this;
    }

    template <typename T>
    complex& operator+=(const std::complex<T>& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    template <typename T>
    complex& operator-=(const std::complex<T>& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    template <typename T>
    complex& operator*=(const std::complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    template <typename T>
    complex& operator/=(const std::complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    complex& operator+=(const complex& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    complex& operator-=(const complex& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    complex& operator*=(const complex& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    complex& operator/=(const complex& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

#define COMPLEX_BFLOAT16_OPERATOR(_op, _opeq)                           \
    friend complex operator _op(const complex& lhf, const complex& rhf) \
    {                                                                   \
        auto a = lhf;                                                   \
        a _opeq rhf;                                                    \
        return a;                                                       \
    }

    COMPLEX_BFLOAT16_OPERATOR(+, +=)
    COMPLEX_BFLOAT16_OPERATOR(-, -=)
    COMPLEX_BFLOAT16_OPERATOR(*, *=)
    COMPLEX_BFLOAT16_OPERATOR(/, /=)

#undef COMPLEX_BFLOAT16_OPERATOR

// before 2024.2, the operation from sycl bfloat16 is too general such that the
// operation between complex and real will use bfloat16 operation. Thus, we need
// to provide here with higher priority in lookup.
#define REAL_BFLOAT16_OPERATOR(_op, _opeq)                                 \
    friend complex operator _op(const complex& lhf, const value_type& rhf) \
    {                                                                      \
        auto a = lhf;                                                      \
        a _opeq rhf;                                                       \
        return a;                                                          \
    }                                                                      \
    friend complex operator _op(const value_type& lhf, const complex& rhf) \
    {                                                                      \
        complex a = lhf;                                                   \
        a _opeq rhf;                                                       \
        return a;                                                          \
    }

    REAL_BFLOAT16_OPERATOR(+, +=)
    REAL_BFLOAT16_OPERATOR(-, -=)
    REAL_BFLOAT16_OPERATOR(*, *=)
    REAL_BFLOAT16_OPERATOR(/, /=)

#undef REAL_BFLOAT16_OPERATOR

    complex operator-() const { return complex(-real_, -imag_); }

private:
    value_type real_;
    value_type imag_;
};


}  // namespace gko


#endif  // GKO_DPCPP_BASE_COMPLEX_HPP_
