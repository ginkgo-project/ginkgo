// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_BASE_EVENT_HPP_
#define GKO_CUDA_BASE_EVENT_HPP_

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

namespace gko {


class CudaEvent : public Event {
public:
    CudaEvent(std::shared_ptr<const gko::CudaExecutor> exec) : exec_(exec)
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventCreate(&event_));
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventRecord(event_, exec->get_stream()));
    }

    ~CudaEvent()
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventDestroy(event_));
    }

    void synchronize() const override
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_CUDA_ERRORS(cudaEventSynchronize(event_));
    }

private:
    std::shared_ptr<const CudaExecutor> exec_;
    cudaEvent_t event_;
};


}  // namespace gko


#endif  // GKO_CUDA_BASE_EVENT_HPP_
