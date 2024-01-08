// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>


int main()
{
    // Use the queue property `in_order` which is DPC++ only
    sycl::queue myQueue{sycl::property::queue::in_order{}};
    return 0;
}
