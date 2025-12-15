// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/event_kernels.hpp"

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace detail {


/**
 * It records the event when constructing the object and destroys the event with
 * object destructor.
 *
 * Note: from the profiler, the destroy needs 1 us, which should not have impact
 * on the performance.
 */
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


}  // namespace detail


namespace kernels {
namespace cuda {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const detail::Event>& event)
{
    event = std::make_shared<detail::CudaEvent>(exec);
}


}  // namespace event
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
