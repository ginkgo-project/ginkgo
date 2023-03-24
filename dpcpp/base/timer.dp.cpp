#include <ginkgo/core/base/timer.hpp>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


DpcppTimer::DpcppTimer(std::shared_ptr<const DpcppExecutor> exec)
    : exec_{std::move(exec)}
{
    if (!exec_->get_queue()
             ->template has_property<
                 sycl::property::queue::enable_profiling>()) {
        GKO_NOT_SUPPORTED(exec_);
    }
}


time_point DpcppTimer::create_time_point()
{
    time_point result;
    result.type_ = time_point::type::dpcpp;
    result.data_.dpcpp_event = new sycl::event{};
    return result;
}


void DpcppTimer::record(time_point& time)
{
    GKO_ASSERT(time.type_ == time_point::type::dpcpp);
    *time.data_.dpcpp_event =
        exec_->get_queue()->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(1, [=](sycl::id<1> id) {});
        });
}


int64 DpcppTimer::difference(const time_point& start, const time_point& stop)
{
    GKO_ASSERT(start.type_ == time_point::type::dpcpp);
    GKO_ASSERT(stop.type_ == time_point::type::dpcpp);
    stop.data_.dpcpp_event->wait_and_throw();
    auto stop_time =
        stop.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto start_time =
        start.get_profiling_info<sycl::info::event_profiling::command_end>();
    return static_cast<int64>(stop_time - start_time);
}


}  // namespace gko