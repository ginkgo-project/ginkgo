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

    // GKO_REGISTER_OPERATION expands to a generic lambda that nvcc rejects in
    // a .cu translation unit, and this test is compiled as one, so the backend
    // kernel is called directly.
    std::shared_ptr<const gko::detail::Event> record_event_on(
        std::shared_ptr<const gko::EXEC_TYPE> e)
    {
        std::shared_ptr<const gko::detail::Event> ev;
#ifdef GKO_COMPILING_CUDA
        gko::kernels::cuda::event::record_event(e, ev);
#elif defined(GKO_COMPILING_HIP)
        gko::kernels::hip::event::record_event(e, ev);
#endif
        return ev;
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


// wait_event must return promptly with work outstanding, where synchronize
// must not. An ordering test would be a race, and a race the implementation
// happens to win proves nothing.
TEST_F(Stream, WaitEventDoesNotBlockTheHost)
{
    // the fixture's value_type, so a single-precision build does not pull
    // in a double instantiation it may not have
    using Dense = gko::matrix::Dense<value_type>;
    // large enough that the device, not the host enqueueing the calls, is
    // the bottleneck
    const gko::size_type n = 1 << 25;
    auto reader = this->make_second_executor(true);
    auto x = Dense::create(this->exec, gko::dim<2>{n, 1});
    auto y = Dense::create(this->exec, gko::dim<2>{n, 1});
    auto alpha = gko::initialize<Dense>({gko::one<value_type>()}, this->exec);
    x->fill(1.0);
    y->fill(1.0);
    this->exec->synchronize();
    // enough queued work that a host-side wait on it is unmistakable
    for (int i = 0; i < 200; i++) {
        x->add_scaled(alpha, y);
    }
    auto ev = this->record_event_on(this->exec);

    const auto t0 = std::chrono::steady_clock::now();
    gko::kernels::GKO_DEVICE_NAMESPACE::event::wait_event(reader, ev.get());
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


// The reference executor has no queue of its own, so waiting on it is a
// host-side synchronize.
TEST_F(Stream, WaitEventOnHostExecutorSynchronizes)
{
    const gko::size_type size = 1 << 20;
    gko::array<int> source{this->ref, size};
    std::fill(source.get_data(), source.get_data() + size, 3);
    gko::array<int> on_device{this->exec, source};
    auto ev = this->record_event_on(this->exec);

    ASSERT_NO_THROW(
        gko::kernels::reference::event::wait_event(this->ref, ev.get()));

    gko::array<int> back{this->ref, on_device};
    GKO_ASSERT_ARRAY_EQ(source, back);
}


}  // namespace
