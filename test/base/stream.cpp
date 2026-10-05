// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include <chrono>
#include <memory>
#include <numeric>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/event_kernels.hpp"
#include "core/test/utils/assertions.hpp"
#include "test/utils/common_fixture.hpp"


namespace {


GKO_REGISTER_OPERATION(record_event, event::record_event);


class Stream : public CommonTestFixture {
protected:
    // A second executor on the same device with a stream of its own; memory
    // is addressable from either.
    std::shared_ptr<gko::EXEC_TYPE> make_second_executor(bool non_blocking)
    {
#ifdef GKO_COMPILING_CUDA
        second_stream = std::make_unique<gko::cuda_stream>(
            ResourceEnvironment::cuda_device_id, non_blocking);
        return gko::CudaExecutor::create(
            ResourceEnvironment::cuda_device_id, ref,
            std::make_shared<gko::CudaAllocator>(), second_stream->get());
#elif defined(GKO_COMPILING_HIP)
        second_stream = std::make_unique<gko::hip_stream>(
            ResourceEnvironment::hip_device_id, non_blocking);
        return gko::HipExecutor::create(ResourceEnvironment::hip_device_id, ref,
                                        std::make_shared<gko::HipAllocator>(),
                                        second_stream->get());
#else
        // only CUDA and HIP have stream wrappers
        return nullptr;
#endif
    }

    void SetUp() override
    {
#if !defined(GKO_COMPILING_CUDA) && !defined(GKO_COMPILING_HIP)
        GTEST_SKIP() << "streams are only exposed for CUDA and HIP";
#endif
    }

#ifdef GKO_COMPILING_CUDA
    std::unique_ptr<gko::cuda_stream> second_stream;
#elif defined(GKO_COMPILING_HIP)
    std::unique_ptr<gko::hip_stream> second_stream;
#endif
};


// A smoke test: the non-blocking semantics are not observable through
// Ginkgo's API, so this only checks the executor works.
TEST_F(Stream, NonBlockingStreamIsUsable)
{
    auto other = this->make_second_executor(true);
    gko::array<int> host{this->ref, 1024};
    std::iota(host.get_data(), host.get_data() + host.get_size(), 0);
    gko::array<int> on_other{other, host};

    gko::array<int> back{this->ref, on_other};

    GKO_ASSERT_ARRAY_EQ(host, back);
}


// stream_wait must return promptly with work outstanding, where synchronize
// must not. An ordering test would be a race, and a race the implementation
// happens to win proves nothing.
TEST_F(Stream, StreamWaitDoesNotBlockTheHost)
{
    using Dense = gko::matrix::Dense<double>;
    // large enough that the device, not the host enqueueing the calls, is
    // the bottleneck
    const gko::size_type n = 1 << 25;
    auto reader = this->make_second_executor(true);
    auto x = Dense::create(this->exec, gko::dim<2>{n, 1});
    auto y = Dense::create(this->exec, gko::dim<2>{n, 1});
    auto alpha = gko::initialize<Dense>({1.0}, this->exec);
    x->fill(1.0);
    y->fill(1.0);
    this->exec->synchronize();
    // enough queued work that a host-side wait on it is unmistakable
    for (int i = 0; i < 200; i++) {
        x->add_scaled(alpha, y);
    }
    std::shared_ptr<const gko::detail::Event> ev;
    this->exec->run(make_record_event(ev));

    const auto t0 = std::chrono::steady_clock::now();
    ev->stream_wait(reader);
    const auto t1 = std::chrono::steady_clock::now();
    ev->synchronize();
    const auto t2 = std::chrono::steady_clock::now();
    const auto waited = std::chrono::duration<double>(t1 - t0).count();
    const auto synced = std::chrono::duration<double>(t2 - t1).count();

    if (synced < 1e-3) {
        GTEST_SKIP() << "the queued work finished in " << synced
                     << " s, too quickly to tell a wait from a "
                        "synchronize";
    }
    ASSERT_LT(waited, synced / 10.0);
}


// The default implementation is a host-side synchronize, checked on the
// reference executor, which has no queue of its own.
TEST_F(Stream, StreamWaitFallsBackToSynchronize)
{
    const gko::size_type size = 1 << 20;
    gko::array<int> source{this->ref, size};
    std::fill(source.get_data(), source.get_data() + size, 3);
    gko::array<int> on_device{this->exec, source};
    std::shared_ptr<const gko::detail::Event> ev;
    this->exec->run(make_record_event(ev));

    ASSERT_NO_THROW(ev->stream_wait(this->ref));

    gko::array<int> back{this->ref, on_device};
    GKO_ASSERT_ARRAY_EQ(source, back);
}


}  // namespace
