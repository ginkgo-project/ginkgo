#include <ginkgo/core/base/timer.hpp>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


HipTimer::HipTimer(std::shared_ptr<const HipExecutor> exec)
    : exec_{std::move(exec)}
{}


time_point HipTimer::create_time_point()
{
    time_point result;
    result.type_ = time_point::type::hip;
    GKO_ASSERT_NO_HIP_ERRORS(hipEventCreate(&result.data_.hip_event));
}


void HipTimer::record(time_point& time)
{
    GKO_ASSERT(time.type_ == time_point::type::hip);
    GKO_ASSERT_NO_HIP_ERRORS(
        hipEventRecord(time.data_.hip_event, exec_->get_stream()));
}


int64 HipTimer::difference(const time_point& start, const time_point& stop)
{
    GKO_ASSERT(start.type_ == time_point::type::hip);
    GKO_ASSERT(stop.type_ == time_point::type::hip);
    GKO_ASSERT_NO_HIP_ERRORS(hipEventSynchronize(stop.data_.hip_event));
    float ms{};
    GKO_ASSERT_NO_HIP_ERRORS(
        hipEventElapsedTime(&ms, start.data_.hip_event, stop.data_.hip_event));
    return static_cast<int64>(ms * double{1e6});
}


}  // namespace gko