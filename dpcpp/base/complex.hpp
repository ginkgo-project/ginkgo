// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_COMPLEX_HPP_
#define GKO_DPCPP_BASE_COMPLEX_HPP_

#include <sycl/half_type.hpp>

#include <ginkgo/config.hpp>

// this file is to workaround for the intel sycl complex different loading.
// intel sycl provides complex and the corresponding searching path. When users
// load complex with -fsycl, the compiler will load intel's <complex> header
// first and then load usual <complex> header. However, it implicitly
// instantiates and uses std::complex<sycl::half>, so we need to provide the
// implementation before that. In ginkgo, we will definitely load <complex> in
// the public interface, which is before sycl backend, so we have no normal way
// to provide the std::complex<sycl::half> implementation in sycl.
// We apply the same trick to load this file first and then load their header
// later. We will also configure this file as <complex> and provide the search
// path in sycl module.
// They start to do this from LIBSYCL 7.1.0.

namespace std {

template <typename>
class complex;

// implement std::complex<sycl::half> before knowing std::complex<float>
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
    complex(const complex<T>& other)
        : real_(static_cast<value_type>(other.real())),
          imag_(static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }

    inline operator std::complex<float>() const noexcept;

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
    complex& operator+=(const complex<T>& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }

    template <typename T>
    complex& operator-=(const complex<T>& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }

    template <typename T>
    inline complex& operator*=(const complex<T>& val);

    template <typename T>
    inline complex& operator/=(const complex<T>& val);

// It's for MacOS.
// TODO: check whether mac compiler always use complex version even when real
// half
#define COMPLEX_HALF_OPERATOR(_op, _opeq)                                  \
    friend complex<sycl::half> operator _op(const complex<sycl::half> lhf, \
                                            const complex<sycl::half> rhf) \
    {                                                                      \
        auto a = lhf;                                                      \
        a _opeq rhf;                                                       \
        return a;                                                          \
    }

    COMPLEX_HALF_OPERATOR(+, +=)
    COMPLEX_HALF_OPERATOR(-, -=)
    COMPLEX_HALF_OPERATOR(*, *=)
    COMPLEX_HALF_OPERATOR(/, /=)

#undef COMPLEX_HALF_OPERATOR

private:
    value_type real_;
    value_type imag_;
};

}  // namespace std


// after providing std::complex<sycl::half>, we can load their <complex> to
// complete the header chain.

#if GINKGO_DPCPP_MAJOR_VERSION > 7 || \
    (GINKGO_DPCPP_MAJOR_VERSION == 7 && GINKGO_DPCPP_MINOR_VERSION >= 1)

#if defined(__has_include_next)
// GCC/clang support go through this path.
#include_next <complex>
#else
// MSVC doesn't support "#include_next", so we take the same workaround in
// stl_wrappers/complex.
#include <../stl_wrappers/complex>
#endif

#else


#include <complex>


#endif


// we know the complex<float> now, so we implement those functions requiring
// complex<float>
namespace std {


inline complex<sycl::half>::operator complex<float>() const noexcept
{
    return std::complex<float>(static_cast<float>(real_),
                               static_cast<float>(imag_));
}


template <typename T>
inline complex<sycl::half>& complex<sycl::half>::operator*=(
    const complex<T>& val)
{
    auto val_f = static_cast<std::complex<float>>(val);
    auto result_f = static_cast<std::complex<float>>(*this);
    result_f *= val_f;
    real_ = result_f.real();
    imag_ = result_f.imag();
    return *this;
}


template <typename T>
inline complex<sycl::half>& complex<sycl::half>::operator/=(
    const complex<T>& val)
{
    auto val_f = static_cast<std::complex<float>>(val);
    auto result_f = static_cast<std::complex<float>>(*this);
    result_f /= val_f;
    real_ = result_f.real();
    imag_ = result_f.imag();
    return *this;
}


}  // namespace std


#endif  // GKO_DPCPP_BASE_COMPLEX_HPP_
