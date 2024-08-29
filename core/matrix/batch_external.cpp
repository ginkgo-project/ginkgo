// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_external.hpp>


namespace gko {
namespace batch {
namespace matrix {


template <typename ValueType>
std::unique_ptr<External<ValueType>> External<ValueType>::create(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
    External::functor_operation<external_apply::simple_type> simple_apply,
    External::functor_operation<external_apply::advanced_type> advanced_apply,
    void* payload)
{
    return std::unique_ptr<External>(new External(
        std::move(exec), size, simple_apply, advanced_apply, payload));
}


template <typename ValueType>
External<ValueType>::External(std::shared_ptr<const Executor> exec)
    : EnableBatchLinOp<External<ValueType>>(std::move(exec))
{}


template <typename ValueType>
External<ValueType>::External(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
    External::functor_operation<external_apply::simple_type> simple_apply,
    External::functor_operation<external_apply::advanced_type> advanced_apply,
    void* payload)
    : EnableBatchLinOp<External<ValueType>>(std::move(exec), size),
      simple_apply_(simple_apply),
      advanced_apply_(advanced_apply),
      payload_(payload)
{
    auto exec_ = this->get_executor();
    if (dynamic_cast<const OmpExecutor*>(exec_.get())) {
        GKO_THROW_IF_INVALID(
            simple_apply_.cpu_apply && advanced_apply_.cpu_apply,
            "cpu_apply needs to be set when "
            "running on ReferenceExecutor or OmpExecutor");
    } else if (dynamic_cast<const CudaExecutor*>(exec_.get())) {
        GKO_THROW_IF_INVALID(
            simple_apply_.cuda_apply && advanced_apply_.cuda_apply,
            "cuda_apply needs to be set when running on CudaExecutor");
    } else if (dynamic_cast<const HipExecutor*>(exec_.get())) {
        GKO_THROW_IF_INVALID(
            simple_apply_.hip_apply && advanced_apply_.hip_apply,
            "hip_apply needs to be set when running on HipExecutor");
    } else if (dynamic_cast<const DpcppExecutor*>(exec_.get())) {
        GKO_THROW_IF_INVALID(
            simple_apply_.sycl_apply && advanced_apply_.sycl_apply,
            "sycl_apply needs to be set when running on DpcppExecutor");
    }
}


#define GKO_DECLARE_BATCH_MATRIX_EXTERNAL(_vtype) class External<_vtype>

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MATRIX_EXTERNAL);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
