// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_BF16_ALIAS_HPP_
#define GKO_DPCPP_BASE_BF16_ALIAS_HPP_

#include <sycl/ext/oneapi/bfloat16.hpp>

namespace gko {


using vendor_bf16 = sycl::ext::oneapi::bfloat16;


}


#endif  // GKO_DPCPP_BASE_BF16_ALIAS_HPP_
