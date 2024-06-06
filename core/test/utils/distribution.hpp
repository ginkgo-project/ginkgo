// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GINKGO_DISTRIBUTION_HPP
#define GINKGO_DISTRIBUTION_HPP

#include <algorithm>
#include <random>


#include <ginkgo/core/base/half.hpp>


namespace gko {
namespace test {
namespace detail {


template <typename Dist>
struct dist_wrapper_half {
    using result_type = gko::half;
    using underlying_dist_type = Dist;
    using underlying_result_type = typename underlying_dist_type::result_type;
    using param_type = typename underlying_dist_type::param_type;

    dist_wrapper_half() = default;

    template <typename... Args,
              std::enable_if_t<
                  std::is_constructible<underlying_dist_type, Args...>::value,
                  int> = 0>
    explicit dist_wrapper_half(Args&&... args)
        : dist_(std::forward<Args>(args)...)
    {}

    explicit dist_wrapper_half(const param_type& p) : dist_(p) {}

    void reset() { dist_.reset(); }

    param_type param() const { return dist_.param(); }

    void param(const param_type& p) { dist_.param(p); }

    template <typename Engine>
    result_type operator()(Engine& e)
    {
        return clamp(dist_(e));
    }

    template <typename Engine>
    result_type operator()(Engine& e, const param_type& p)
    {
        return clamp(dist_(e, p));
    }

    result_type min() const { return clamp(dist_.min()); }

    result_type max() const { return clamp(dist_.max()); }

    friend bool operator==(const dist_wrapper_half& lhs,
                           const dist_wrapper_half& rhs)
    {
        return lhs.dist_ == rhs.dist_;
    }

    friend bool operator!=(const dist_wrapper_half& lhs,
                           const dist_wrapper_half& rhs)
    {
        return lhs.dist_ != rhs.dist_;
    }

    template <class CharT, class Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(
        std::basic_ostream<CharT, Traits>& ost, const dist_wrapper_half& d)
    {
        return ost << d.dist_;
    }

    template <class CharT, class Traits>
    friend std::basic_ostream<CharT, Traits>& operator>>(
        std::basic_istream<CharT, Traits>& ist, const dist_wrapper_half& d)
    {
        return ist >> d.dist_;
    }

private:
    constexpr const underlying_result_type& clamp(
        const underlying_result_type& v) const
    {
        return std::clamp(v, std::numeric_limits<result_type>::lowest(),
                          std::numeric_limits<result_type>::max());
    }

    underlying_dist_type dist_;
};


template <typename ValueType>
struct normal_distribution {
    using type = std::normal_distribution<ValueType>;
};

template <>
struct normal_distribution<gko::half> {
    using type = dist_wrapper_half<std::normal_distribution<float>>;
};


template <typename ValueType>
struct uniform_real_distribution {
    using type = std::uniform_real_distribution<ValueType>;
};

template <>
struct uniform_real_distribution<gko::half> {
    using type = dist_wrapper_half<std::uniform_real_distribution<float>>;
};


}  // namespace detail


template <typename ValueType = double>
using normal_distribution =
    typename detail::normal_distribution<ValueType>::type;


template <typename ValueType = double>
using uniform_real_distribution =
    typename detail::uniform_real_distribution<ValueType>::type;


}  // namespace test
}  // namespace gko


#endif  // GINKGO_DISTRIBUTION_HPP
