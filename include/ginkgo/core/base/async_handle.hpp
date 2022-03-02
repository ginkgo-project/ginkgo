/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_BASE_ASYNC_HANDLE_HPP_
#define GKO_PUBLIC_CORE_BASE_ASYNC_HANDLE_HPP_


#include <array>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/device.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/machine_topology.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/log/logger.hpp>
#include <ginkgo/core/synthesizer/containers.hpp>


struct CUstream_st;

struct ihipStream_t;


inline namespace cl {
namespace sycl {

class queue;

}  // namespace sycl
}  // namespace cl


namespace gko {


template <typename>
class HostAsyncHandle;


namespace detail {


template <typename>
class AsyncHandleBase;


}  // namespace detail


class Operation;


class AsyncHandle {
    template <typename T>
    friend class detail::AsyncHandleBase;
    template <typename T>
    friend class HostAsyncHandle;

public:
    virtual ~AsyncHandle() = default;

    AsyncHandle() = default;
    AsyncHandle(AsyncHandle&) = delete;
    AsyncHandle(AsyncHandle&&) = default;
    AsyncHandle& operator=(AsyncHandle&) = delete;
    AsyncHandle& operator=(AsyncHandle&&) = default;

    virtual void wait() = 0;

    virtual void wait_for(const std::chrono::duration<int>& time) = 0;

    virtual void wait_until(
        const std::chrono::time_point<std::chrono::steady_clock>& time) = 0;

    virtual std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) = 0;
};


namespace detail {


template <typename ConcreteAsyncHandle>
class AsyncHandleBase : public AsyncHandle {
    template <typename T>
    friend class HostAsyncHandle;

public:
    void wait() override { self()->wait(); }

    void wait_for(const std::chrono::duration<int>& time) override
    {
        self()->wait_for(time);
    }

    void wait_until(
        const std::chrono::time_point<std::chrono::steady_clock>& time) override
    {
        self()->wait_until(time);
    }

    std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) override
    {
        return self()->then(handle);
    }

private:
    ConcreteAsyncHandle* self() noexcept
    {
        return static_cast<ConcreteAsyncHandle*>(this);
    }

    const ConcreteAsyncHandle* self() const noexcept
    {
        return static_cast<const ConcreteAsyncHandle*>(this);
    }
};


}  // namespace detail


template <typename T = void>
class HostAsyncHandle
    : public detail::AsyncHandleBase<HostAsyncHandle<T>>,
      public std::enable_shared_from_this<HostAsyncHandle<T>> {
    friend class detail::AsyncHandleBase<HostAsyncHandle<T>>;

public:
    static std::shared_ptr<HostAsyncHandle> create()
    {
        return std::shared_ptr<HostAsyncHandle>(new HostAsyncHandle());
    }

    static std::shared_ptr<HostAsyncHandle> create(std::future<T> handle)
    {
        return std::shared_ptr<HostAsyncHandle>(
            new HostAsyncHandle(std::move(handle)));
    }

    std::future<T> get_handle() { return &this->handle_; }

    void get_result() { this->handle_.get(); }

    void wait() override { this->handle_.wait(); }

    void wait_for(const std::chrono::duration<int>& time) override
    {
        this->handle_.wait_for(time);
    }

    void wait_until(
        const std::chrono::time_point<std::chrono::steady_clock>& time) override
    {
        this->handle_.wait_until(time);
    }

    std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) override
        GKO_NOT_IMPLEMENTED;

    template <typename Closure>
    std::shared_ptr<AsyncHandle> queue(const Closure& op) GKO_NOT_IMPLEMENTED;

    std::shared_ptr<AsyncHandle> queue(std::future<T> new_op)
        GKO_NOT_IMPLEMENTED;

protected:
    HostAsyncHandle() : handle_() {}

    HostAsyncHandle(std::future<T> input_handle)
        : handle_(std::move(input_handle))
    {}

private:
    std::future<T> handle_;
};


class CudaAsyncHandle : public detail::AsyncHandleBase<CudaAsyncHandle>,
                        public std::enable_shared_from_this<CudaAsyncHandle> {
    friend class detail::AsyncHandleBase<CudaAsyncHandle>;

public:
    enum class create_type { non_blocking, legacy_blocking, default_blocking };

    static std::shared_ptr<CudaAsyncHandle> create(
        create_type c_type = create_type::legacy_blocking)
    {
        return std::shared_ptr<CudaAsyncHandle>(new CudaAsyncHandle(c_type));
    }

    static std::shared_ptr<CudaAsyncHandle> create(CUstream_st* handle)
    {
        return std::shared_ptr<CudaAsyncHandle>(
            new CudaAsyncHandle(std::move(handle)));
    }

    CUstream_st* get_handle() { return this->handle_.get(); }

    void get_result();

    void wait() override;

    void wait_for(const std::chrono::duration<int>& time) override;

    void wait_until(const std::chrono::time_point<std::chrono::steady_clock>&
                        time) override;

    std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) override
        GKO_NOT_IMPLEMENTED;

    template <typename Closure>
    std::shared_ptr<AsyncHandle> queue(const Closure& op) GKO_NOT_IMPLEMENTED;

protected:
    CudaAsyncHandle(create_type c_type);

    CudaAsyncHandle(CUstream_st* input_handle)
        : handle_(std::move(input_handle))
    {}

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<CUstream_st> handle_;
};


class HipAsyncHandle : public detail::AsyncHandleBase<HipAsyncHandle>,
                       public std::enable_shared_from_this<HipAsyncHandle> {
    friend class detail::AsyncHandleBase<HipAsyncHandle>;

public:
    enum class create_type { non_blocking, legacy_blocking, default_blocking };

    static std::shared_ptr<HipAsyncHandle> create(
        create_type c_type = create_type::default_blocking)
    {
        return std::shared_ptr<HipAsyncHandle>(new HipAsyncHandle(c_type));
    }

    static std::shared_ptr<HipAsyncHandle> create(ihipStream_t* handle)
    {
        return std::shared_ptr<HipAsyncHandle>(
            new HipAsyncHandle(std::move(handle)));
    }

    ihipStream_t* get_handle() { return this->handle_.get(); }

    void get_result();

    void wait() override;

    void wait_for(const std::chrono::duration<int>& time) override;

    void wait_until(const std::chrono::time_point<std::chrono::steady_clock>&
                        time) override;

    std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) override
        GKO_NOT_IMPLEMENTED;

    template <typename Closure>
    std::shared_ptr<AsyncHandle> queue(const Closure& op) GKO_NOT_IMPLEMENTED;

protected:
    HipAsyncHandle(create_type c_type);

    HipAsyncHandle(ihipStream_t* input_handle)
        : handle_(std::move(input_handle))
    {}

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<ihipStream_t> handle_;
};


class DpcppAsyncHandle : public detail::AsyncHandleBase<DpcppAsyncHandle>,
                         public std::enable_shared_from_this<DpcppAsyncHandle> {
    friend class detail::AsyncHandleBase<DpcppAsyncHandle>;

public:
    static std::shared_ptr<DpcppAsyncHandle> create(::cl::sycl::queue* handle)
    {
        return std::shared_ptr<DpcppAsyncHandle>(
            new DpcppAsyncHandle(std::move(handle)));
    }

    ::cl::sycl::queue* get_handle() { return this->handle_.get(); }

    void get_result();

    void wait() override;

    void wait_for(const std::chrono::duration<int>& time) override;

    void wait_until(const std::chrono::time_point<std::chrono::steady_clock>&
                        time) override;

    std::shared_ptr<AsyncHandle> then(AsyncHandle* handle) override
        GKO_NOT_IMPLEMENTED;

    template <typename Closure>
    std::shared_ptr<AsyncHandle> queue(const Closure& op) GKO_NOT_IMPLEMENTED;

protected:
    DpcppAsyncHandle(::cl::sycl::queue* input_handle)
        : handle_(std::move(input_handle))
    {}

private:
    template <typename T>
    using handle_manager = std::unique_ptr<T, std::function<void(T*)>>;
    handle_manager<::cl::sycl::queue> handle_;
};


}  // namespace gko


#endif  // GKO_PUBLIC_CORE_BASE_ASYNC_HANDLE_HPP_
