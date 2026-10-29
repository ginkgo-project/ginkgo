// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/event_kernels.hpp"

#include <memory>

#include <sycl/sycl.hpp>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace detail {


/**
 * It records the event when constructing the object.
 *
 * Note: If it encounters large launch overhead, we might need to keep the last
 * event in Executor and grab it here rather than creating a new one.
 */
class DpcppEvent : public Event {
public:
    DpcppEvent(std::shared_ptr<const gko::DpcppExecutor> exec) : exec_(exec)
    {
        event_ = exec_->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(1, [=](sycl::id<1> id) {});
        });
    }

    ~DpcppEvent() {}

    void synchronize() const override { event_.wait_and_throw(); }

private:
    std::shared_ptr<const DpcppExecutor> exec_;
    // wait_and_throw is not a const function. We use synchrnoize() const to
    // keep the same interface as executor->synchronize()
    mutable sycl::event event_;
};


}  // namespace detail


namespace kernels {
namespace dpcpp {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const detail::Event>& event)
{
    event = std::make_shared<detail::DpcppEvent>(exec);
}


}  // namespace event
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
