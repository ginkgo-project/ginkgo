// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_PRECONDITIONER_BATCH_IDENTITY_HPP_
#define GKO_DPCPP_PRECONDITIONER_BATCH_IDENTITY_HPP_


#include <memory>

#include <CL/sycl.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_preconditioner {


/**
 * @see reference/preconditioner/batch_identity.hpp
 */
template <typename ValueType>
class Identity final {
public:
    using value_type = ValueType;

    static constexpr int work_size = 0;

    static int dynamic_work_size(int, int) { return 0; }

    template <typename batch_item_type>
    void generate(size_type, const batch_item_type&, ValueType*,
                  sycl::nd_item<3> item_ct1)
    {}

    __dpct_inline__ void apply(const int num_rows, const ValueType* const r,
                               ValueType* const z,
                               sycl::nd_item<3> item_ct1) const
    {
        for (int li = item_ct1.get_local_linear_id(); li < num_rows;
             li += item_ct1.get_local_range().size()) {
            z[li] = r[li];
        }
    }
};


}  // namespace batch_preconditioner
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif
