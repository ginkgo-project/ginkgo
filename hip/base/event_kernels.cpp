// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/event_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/cuda_hip/base/runtime.hpp"


namespace gko {
namespace detail {


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
        // a destructor must not throw, and failing to destroy the event only
        // leaks it
        (void)hipEventDestroy(event_);
    }

    void synchronize() const override
    {
        auto guard = exec_->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(event_));
    }

    hipEvent_t get() const { return event_; }

private:
    std::shared_ptr<const HipExecutor> exec_;
    hipEvent_t event_;
};


}  // namespace detail


namespace kernels {
namespace hip {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const detail::Event>& event)
{
    event = std::make_shared<detail::HipEvent>(exec);
}


void wait_event(std::shared_ptr<const DefaultExecutor> exec,
                const detail::Event* event)
{
    // an event from another backend has no handle the stream can wait on
    if (auto hip_event = dynamic_cast<const detail::HipEvent*>(event)) {
        auto guard = exec->get_scoped_device_id_guard();
        GKO_ASSERT_NO_HIP_ERRORS(
            hipStreamWaitEvent(exec->get_stream(), hip_event->get(), 0));
    } else {
        event->synchronize();
    }
}


}  // namespace event
}  // namespace hip
}  // namespace kernels
}  // namespace gko
