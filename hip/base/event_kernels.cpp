// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/event_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/cuda_hip/base/runtime.hpp"


namespace gko {


/**
 * It records the event when constructing the object and destroys the event with
 * object destructor.
 */
class HipEvent : public Event {
public:
    HipEvent(std::shared_ptr<const gko::HipExecutor> exec) : exec_(exec)
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&event_));
        GKO_ASSERT_NO_HIP_ERRORS(hipEventRecord(event_, exec->get_stream()));
    }

    ~HipEvent()
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(hipEventDestroy(event_));
    }

    void synchronize() const override
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(event_));
    }

private:
    std::shared_ptr<const HipExecutor> exec_;
    hipEvent_t event_;
};


namespace kernels {
namespace hip {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const Event>& event)
{
    event = std::make_shared<HipEvent>(exec);
}


}  // namespace event
}  // namespace hip
}  // namespace kernels
}  // namespace gko
