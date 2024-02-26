// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <CL/sycl.hpp>


#include "dpcpp/base/helper.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


bool validate(sycl::queue* queue, unsigned int workgroup_size,
              unsigned int subgroup_size)
{
    auto device = queue->get_device();
    auto subgroup_size_list =
        device.get_info<sycl::info::device::sub_group_sizes>();
    auto max_workgroup_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    bool allowed = false;
    for (auto& i : subgroup_size_list) {
        allowed |= (i == subgroup_size);
    }
    return allowed && (workgroup_size <= max_workgroup_size);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
