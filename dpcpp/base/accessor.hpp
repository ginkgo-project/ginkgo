// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_ACCESSOR_HPP_
#define GKO_DPCPP_BASE_ACCESSOR_HPP_


#include <complex>

#include <sycl/half_type.hpp>

#include <ginkgo/core/base/bfloat16.hpp>
#include <ginkgo/core/base/half.hpp>

#include "accessor/sycl_helper.hpp"
#include "dpcpp/base/bf16_alias.hpp"
#include "dpcpp/base/complex.hpp"


namespace acc {


template <>
struct sycl_type<gko::half> {
    using type = sycl::half;
};

template <>
struct sycl_type<gko::bfloat16> {
    using type = gko::vendor_bf16;
};

template <>
struct sycl_type<std::complex<gko::half>> {
    using type = gko::complex<typename sycl_type<gko::half>::type>;
};

template <>
struct sycl_type<std::complex<gko::bfloat16>> {
    using type = gko::complex<typename sycl_type<gko::bfloat16>::type>;
};


}  // namespace acc


#endif  // GKO_DPCPP_BASE_ACCESSOR_HPP_
